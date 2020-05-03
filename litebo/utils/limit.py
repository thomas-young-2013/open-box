import sys
from contextlib import contextmanager
if sys.platform != 'win32':
    import signal


def get_platform():
    platforms = {
        'linux1': 'Linux',
        'linux2': 'Linux',
        'darwin': 'OS X',
        'win32': 'Windows'
    }
    if sys.platform not in platforms:
        raise ValueError('Unsupported OS: %s' % sys.platform)

    return platforms[sys.platform]


class TimeoutException(Exception):
    pass


"""
https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call-in-python
"""


@contextmanager
def time_limit(seconds):
    skip_flag = False if sys.platform == 'win32' else True

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    if skip_flag:
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
    try:
        yield
    finally:
        if skip_flag:
            signal.alarm(0)
