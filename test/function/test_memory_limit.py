import os
import sys
sys.path.append(os.getcwd())
from litebo.utils.limit_ import limit_function


def test_func(*args, **kwargs):
    m = [10000] * 1024 * 1024 * 10
    print('matrix size in megabytes', sys.getsizeof(m) / 1024 / 1024)
    return 12


if __name__ == "__main__":
    a = (3,)
    b = dict()
    res = limit_function(test_func, 20, 200, a, b)
    print(res)
