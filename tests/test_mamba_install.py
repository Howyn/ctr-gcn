import importlib.util
from pathlib import Path
import unittest


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


if __name__ == '__main__':
    unittest.main()
