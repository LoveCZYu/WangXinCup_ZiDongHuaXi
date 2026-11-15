"""Regression: official EXE must support a noninteractive, redirected launch."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from exe_runtime import decode_console
from launch_server import TerminalRuntime


@contextmanager
def checked_runtime(logs):
    try:
        with TerminalRuntime(logs) as runtime:
            yield runtime
    except subprocess.CalledProcessError as error:
        path = logs / 'exe-runtime.log'
        detail = decode_console(path.read_bytes()) if path.exists() else '(no runtime log)'
        raise AssertionError(f'Wine initialization failed: {error}\n{detail}') from error


class WineRedirectTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux', 'Linux Wine compatibility')
    def test_official_exe_help_with_closed_stdin_and_redirected_stdout(self):
        status = subprocess.check_output(['dpkg-query', '-W', '-f=${Status}', 'wine32:i386'], text=True)
        self.assertEqual(status, 'install ok installed', 'Official 11.15 runtime requires Wine32')
        with tempfile.TemporaryDirectory() as folder:
            logs = Path(folder)
            with checked_runtime(logs) as runtime:
                command = runtime.command(ROOT / 'data/official/serve.exe', ['--help'])
                with (logs / 'help.log').open('wb') as output:
                    result = subprocess.run(command, env=runtime.env, stdin=subprocess.DEVNULL,
                                            stdout=output, stderr=subprocess.STDOUT, timeout=30)
                text = decode_console((logs / 'help.log').read_bytes())
                self.assertEqual(result.returncode, 0, text)
                self.assertIn('--config', text)
                self.assertIn('收发程序', text)
                self.assertNotIn('WinError 6', text)


if __name__ == '__main__':
    unittest.main()
