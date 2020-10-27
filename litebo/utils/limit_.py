import sys
import dill
import time
import psutil
from collections import namedtuple
from multiprocessing import Process, Manager, freeze_support, Pipe


class SignalException(Exception):
    pass


class TimeoutException(Exception):
    pass


class OutOfMemoryLimitException(Exception):
    pass


def get_platform():
    platforms = {
        'linux': 'Linux',
        'linux1': 'Linux',
        'linux2': 'Linux',
        'darwin': 'OSX',
        'win32': 'Windows'
    }
    if sys.platform not in platforms:
        raise ValueError('Unsupported platform - %s.' % sys.platform)
    return platforms[sys.platform]


_platform = get_platform()
if _platform not in ['win32']:
    import resource
Returns = namedtuple('return_values', ['status', 'result'])


def wrapper(*args, **kwargs):
    # Parse args.
    _func, _conn, _time_limit, _mem_limit, args = args[0], args[1], args[2], args[3], args[4:]
    _func = dill.loads(_func)
    result = (False, None)

    if _platform in ['Linux', 'OSX']:
        import signal

        def handler(signum, frame):
            if signum == signal.SIGALRM:
                raise TimeoutException
            else:
                raise SignalException

        # Limit the memory usage.
        if _mem_limit is not None:
            # Transform megabyte to byte
            mem_in_b = _mem_limit * 1024 * 1024

            # Set the maximum size (in bytes) of address space.
            resource.setrlimit(resource.RLIMIT_AS, (mem_in_b, mem_in_b))

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(_time_limit)
    try:
        print('start to call the function.')
        result = (True, _func(*args, **kwargs))
        print('calling ends.')
    except TimeoutException:
        result = (False, TimeoutException)
    except MemoryError:
        result = (False, OutOfMemoryLimitException)
    except SignalException:
        result = (False, SignalException)

    finally:
        try:
            _conn.send(result)
            _conn.close()
        except:
            pass
        finally:
            p = psutil.Process()
            for child in p.children(recursive=True):
                child.kill()


def limit_function(func, wall_clock_time, mem_usage_limit, *args, **kwargs):
    """
    :param func: the objective function to call.
    :param wall_clock_time: seconds.
    :param mem_usage_limit: megabytes.
    :param args:
    :param kwargs:
    :return:
    """

    if _platform == 'Windows':
        freeze_support()
    parent_conn, child_conn = Pipe(False)

    # Deal with special case in Bayesian optimization.
    if len(args) == 0 and 'args' in kwargs:
        args = kwargs['args']
        kwargs = kwargs['kwargs']

    func = dill.dumps(func)
    args = [func] + [child_conn] + [wall_clock_time] + [mem_usage_limit] + list(args)

    p = Process(target=wrapper, args=tuple(args), kwargs=kwargs)
    p.start()
    # Special case on windows.
    if _platform in ['Windows']:
        p_id = p.pid
        exceed_mem_limit = False
        start_time = time.time()
        while time.time() <= start_time + wall_clock_time:
            mem_used = psutil.Process(p_id).memory_info().vms / 1024 / 1024
            if mem_used > mem_usage_limit:
                exceed_mem_limit = True
                break
            time.sleep(1.)

        if exceed_mem_limit:
            p.terminate()
            return Returns(status=False, result=OutOfMemoryLimitException)
        if p.is_alive():
            p.terminate()
            return Returns(status=False, result=TimeoutException)
        result = parent_conn.recv()
        parent_conn.close()
        return result
    else:
        p.join(wall_clock_time)
        if p.is_alive():
            p.terminate()
            return Returns(status=False, result=TimeoutException)
        result = parent_conn.recv()
        parent_conn.close()
        return result
