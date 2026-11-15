"""Linux launch compatibility layer around the unchanged official server."""
import signal
import shlex
import sys

from exe_runtime import ExeRuntime
import serve

class TerminalRuntime(ExeRuntime):
    def command(self, executable, arguments):
        command = super().command(executable, arguments)
        if sys.platform != 'linux':
            return command
        # The bundled Python EXE fails init_sys_streams with redirected handles.
        # script provides a PTY and relays its output to the official server log.
        # C.UTF-8 exists in the base image; zh_CN.UTF-8 does not.
        return ['env', 'LANG=C.UTF-8', 'LC_ALL=C.UTF-8', 'script', '-q', '-e', '-f',
                '-c', shlex.join(command), '/dev/null']


if __name__ == '__main__':
    serve.ExeRuntime = TerminalRuntime
    signal.signal(signal.SIGTERM, serve.interrupted)
    sys.exit(serve.main())
