import sys
from contextlib import contextmanager
if sys.platform != 'win32':
    import signal


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
