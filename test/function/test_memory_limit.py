import os
import sys
sys.path.append(os.getcwd())
from litebo.utils.limit_ import limit_function


def test_func(*args, **kwargs):
    import sys
    m = [10000] * 1024 * 1024 * 10
    print('matrix size in megabytes', sys.getsizeof(m) / 1024 / 1024)
    import numpy as np
    # change mat_n: {10000, 1000, 100, 10}
    mat_n = 100
    m = np.random.random((mat_n, mat_n))
    from sklearn.decomposition import KernelPCA

    for _ in range(100):
        pca = KernelPCA()
        pca.fit_transform(m)

    return 12


if __name__ == "__main__":
    a = (3,)
    b = dict()
    res = limit_function(test_func, 20, 85, a, b)
    print(res)
