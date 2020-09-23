import os
import time
import psutil
from collections import namedtuple
from multiprocessing import Process, Manager
from solnml.components.feature_engineering.transformations.generator.kernel_pca import KernelPCA
from solnml.components.feature_engineering.transformations.rescaler.normalizer import NormalizeTransformation
from solnml.datasets.utils import load_train_test_data

data, _ = load_train_test_data('codrna', data_dir='/Users/thomasyoung/PycharmProjects/soln-ml/', task_type=0)
import pynisher


# @pynisher.enforce_limits(wall_time_in_s=1)
def collect(func):
    def wrapper(*args, **kwargs):
        result_container = args[-1]
        args = tuple(list(args)[:-1])
        result = func(*args, **kwargs)
        result_container.append(result)
        return result_container
    return wrapper


@collect
def kernel_pca(*args, **kwargs):
    kp = KernelPCA()
    norm = NormalizeTransformation()
    _data = norm.operate(data)
    _data = kp.operate(_data)
    print(_data.data[0].shape)
    return 111

start_time = time.time()


def time_limit(func, args, kwargs, time):
    Returns = namedtuple('return_values', ['status', 'results'])
    with Manager() as manager:
        result_container = manager.list()
        args = list(args) + [result_container]
        p = Process(target=func, args=tuple(args), kwargs=kwargs)
        p.start()
        p.join(time)
        result = list(result_container)
    if p.is_alive():
        p.terminate()
        return Returns(status=False, results=None)
    return Returns(status=True, results=result)


def ps_time_limit(func, args, kwargs, time):
    Returns = namedtuple('return_values', ['status', 'results'])
    with Manager() as manager:
        result_container = manager.list()
        args = list(args) + [result_container]
        p1 = Process(target=func, args=tuple(args), kwargs=kwargs)
        p1.start()
        # def on_terminate(proc):
        #     print("process {} terminated with exit code {}".format(proc, proc.returncode))
        procs = psutil.Process().children()
        print(p1.pid)
        print(os.getpid())
        print(procs)
        procs = [p for p in procs if p.pid == p1.pid]
        print(procs)
        gone, alive = psutil.wait_procs(procs, timeout=time)
        result = list(result_container)

    if len(alive) > 0:
        for p in alive:
            p.kill()
        return Returns(status=False, results=None)
    return Returns(status=True, results=result)


a = ()
b = dict()
# res = time_limit(kernel_pca, a, b, 5)
res = ps_time_limit(kernel_pca, a, b, 1)
print(res)
