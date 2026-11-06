import importlib.util
import http.client
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile


spec = importlib.util.spec_from_file_location(
    'install_mamba', Path(__file__).resolve().parents[1] / 'tools' / 'install_mamba.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class WheelSelectionTests(unittest.TestCase):
    def test_existing_autodl_environment_selects_matching_official_wheels(self):
        urls = installer.official_wheel_urls('2.5.1+cu124', '12.4', (3, 12), False)
        self.assertTrue(urls[0].startswith('https://github.com/Dao-AILab/causal-conv1d/'))
        self.assertTrue(urls[1].startswith('https://github.com/state-spaces/mamba/'))
        self.assertTrue(all(url.endswith('cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl')
                            for url in urls))

    def test_abi_flag_is_not_assumed(self):
        urls = installer.official_wheel_urls('2.5.1+cu124', '12.4', (3, 12), True)
        self.assertTrue(all('cxx11abiTRUE-' in url for url in urls))

    def test_other_torch_or_cpu_environment_is_rejected(self):
        for version, cuda in [('2.6.0', '12.4'), ('2.5.1+cpu', None)]:
            with self.assertRaises(ValueError):
                installer.official_wheel_urls(version, cuda, (3, 12), False)


class Response(io.BytesIO):
    def __init__(self, payload, status=200, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = headers or {'Content-Length': str(len(payload))}


class ResumableDownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name)
        self.url = 'https://github.com/example/releases/download/v1/example-1-py3-none-any.whl'
        self.target = self.cache / 'example-1-py3-none-any.whl'
        self.partial = self.target.with_suffix('.whl.part')
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as wheel:
            wheel.writestr('example-1.dist-info/WHEEL', 'Wheel-Version: 1.0\n')
            wheel.writestr('example-1.dist-info/METADATA', 'Name: example\nVersion: 1\n')
            wheel.writestr('example/module.so', 'binary module payload')
        self.payload = archive.getvalue()

    def download(self, attempts=2):
        return installer.download_wheel(self.url, self.cache, attempts=attempts, retry_delay=0)

    def test_short_transfer_resumes_instead_of_installing_a_truncated_file(self):
        def open_response(request, timeout):
            if request.get_header('Range') is None:
                return Response(self.payload[:80], headers={'Content-Length': str(len(self.payload))})
            self.assertEqual(request.get_header('Range'), 'bytes=80-')
            return Response(self.payload[80:], status=206, headers={
                'Content-Range': 'bytes 80-{}/{}'.format(len(self.payload)-1, len(self.payload))})
        with patch.object(installer.urllib.request, 'urlopen', side_effect=open_response) as opened:
            result = self.download()
        self.assertEqual(opened.call_count, 2)
        self.assertEqual(result.read_bytes(), self.payload)
        self.assertFalse(self.partial.exists())

    def test_proxy_ignoring_range_restarts_without_appending_duplicate_bytes(self):
        self.partial.write_bytes(self.payload[:80])
        with patch.object(installer.urllib.request, 'urlopen', return_value=Response(self.payload)) as opened:
            result = self.download()
        self.assertEqual(opened.call_args.args[0].get_header('Range'), 'bytes=80-')
        self.assertEqual(result.read_bytes(), self.payload)

    def test_broken_chunked_response_is_retried(self):
        class BrokenResponse(Response):
            def read(self, size=-1):
                raise http.client.IncompleteRead(b'', 80)
        with patch.object(installer.urllib.request, 'urlopen', side_effect=[
                BrokenResponse(b''), Response(self.payload)]) as opened:
            result = self.download()
        self.assertEqual(opened.call_count, 2)
        self.assertEqual(result.read_bytes(), self.payload)

    def test_server_delivering_a_partial_range_preserves_and_resumes_it(self):
        with patch.object(installer.urllib.request, 'urlopen', side_effect=[
                Response(self.payload[:80], status=206,
                         headers={'Content-Range': 'bytes 0-79/{}'.format(len(self.payload))}),
                Response(self.payload[80:], status=206, headers={
                    'Content-Range': 'bytes 80-{}/{}'.format(len(self.payload)-1, len(self.payload))})
        ]) as opened:
            result = self.download()
        self.assertEqual(opened.call_args_list[1].args[0].get_header('Range'), 'bytes=80-')
        self.assertEqual(result.read_bytes(), self.payload)

    def test_corrupt_payload_is_retried_from_scratch(self):
        corrupt = bytearray(self.payload)
        corrupt[self.payload.index(b'binary module payload')] ^= 1
        with patch.object(installer.urllib.request, 'urlopen', side_effect=[
                Response(corrupt), Response(self.payload)]) as opened:
            result = self.download()
        self.assertIsNone(opened.call_args_list[1].args[0].get_header('Range'))
        self.assertEqual(result.read_bytes(), self.payload)

    def test_valid_cache_is_reused_without_another_network_download(self):
        self.target.write_bytes(self.payload)
        with patch.object(installer.urllib.request, 'urlopen') as opened:
            self.assertEqual(self.download(), self.target)
        opened.assert_not_called()

    def test_html_error_page_is_never_published_as_a_wheel(self):
        with patch.object(installer.urllib.request, 'urlopen', return_value=Response(b'<html>error</html>')):
            with self.assertRaisesRegex(RuntimeError, 'valid wheel'):
                self.download(attempts=1)
        self.assertFalse(self.target.exists())
        self.assertFalse(self.partial.exists())

    def test_retry_exhaustion_preserves_partial_download_for_next_run(self):
        with patch.object(installer.urllib.request, 'urlopen', return_value=Response(
                self.payload[:80], headers={'Content-Length': str(len(self.payload))})):
            with self.assertRaises(RuntimeError):
                self.download(attempts=1)
        self.assertEqual(self.partial.read_bytes(), self.payload[:80])
        self.assertFalse(self.target.exists())

    def test_invalid_cached_wheel_is_replaced(self):
        self.target.write_bytes(b'incomplete wheel')
        with patch.object(installer.urllib.request, 'urlopen', return_value=Response(self.payload)):
            self.download()
        self.assertEqual(self.target.read_bytes(), self.payload)


if __name__ == '__main__':
    unittest.main()
