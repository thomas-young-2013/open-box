<p align="center">
<img src="https://raw.githubusercontent.com/thomas-young-2013/lite-bo/master/docs/logos/logo.png" width="68%">
</p>

-----------

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/thomas-young-2013/lite-bo/blob/master/LICENSE)
[![Build Status](https://api.travis-ci.org/thomas-young-2013/lite-bo.svg?branch=master)](https://api.travis-ci.org/thomas-young-2013)
[![Issues](https://img.shields.io/github/issues-raw/thomas-young-2013/lite-bo.svg)](https://github.com/thomas-young-2013/lite-bo/issues?q=is%3Aissue+is%3Aopen)
[![Bugs](https://img.shields.io/github/issues/thomas-young-2013/lite-bo/bug.svg)](https://github.com/thomas-young-2013/lite-bo/issues?q=is%3Aissue+is%3Aopen+label%3Abug)
[![Pull Requests](https://img.shields.io/github/issues-pr-raw/thomas-young-2013/lite-bo.svg)](https://github.com/thomas-young-2013/lite-bo/pulls?q=is%3Apr+is%3Aopen)
[![Version](https://img.shields.io/github/release/thomas-young-2013/lite-bo.svg)](https://github.com/thomas-young-2013/lite-bo/releases) [![Join the chat at https://gitter.im/bbo-open-box](https://badges.gitter.im/bbo-open-box.svg)](https://gitter.im/bbo-open-box?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Documentation Status](https://readthedocs.org/projects/lite-bo/badge/?version=latest)](https://lite-bo.readthedocs.io/en/latest/?badge=latest)


## OpenBox: Generalized and Efficient Blackbox Optimization System.
OpenBox is an efficient and generalized blackbox optimization (BBO) system, which owns the following characteristics:
1. Basic BBO algorithms.
2. BBO with constraints.
3. BBO with multiple objectives.
4. BBO with transfer learning.
5. BBO with distributed parallelization.
6. BBO with multi-fidelity acceleration.
7. BBO with early stops.


## Deployment Artifacts
#### Standalone Python package.
Users can install the released package and use it using Python.

#### Distributed BBO service.
We adopt the "BBO as a service" paradigm and implement OpenBox as a managed general service for black-box optimization. Users can access this service via REST API conveniently, and do not need to worry about other issues such as environment setup, software maintenance, programming, and optimization of the execution. Moreover, we also provide a Web UI,
through which users can easily track and manage the tasks.


## Features

+ Ease of use. Minimal user configuration and setup, and necessary visualization for optimization process. 
+ Performance standards. Host state-of-the-art optimization algorithms; select proper algorithms automatically.
+ Cost-oriented management. Give cost-model based suggestions to users, e.g., minimal machines or time-budget. 
+ Scalability. Scale to dimensions on the number of input variables, objectives, tasks, trials, and parallel evaluations.
+ High efficiency. Effective use of parallel resource, speeding up optimization with transfer-learning, and multi-fidelity acceleration for computationally-expensive evaluations. 
+ Data privacy protection, robustness and extensibility.

## Links
+ Blog post: [to appear soon]()
+ Documentation: https://lite-bo.readthedocs.io/en/latest/?badge=latest
+ Pypi package: https://pypi.org/project/lite-bo/
+ Conda package: [to appear soon]()
+ Examples: https://github.com/thomas-young-2013/lite-bo/tree/master/examples

## Benchmark Results

Single-objective problems
Ackley-4                  | Hartmann
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/thomas-young-2013/lite-bo/master/docs/experiments/so_math_ackley-4.png)  |  ![](https://raw.githubusercontent.com/thomas-young-2013/lite-bo/master/docs/experiments/so_math_hartmann.png)

Single-objective problems with constraints
Mishra                  | Keane-10
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/thomas-young-2013/lite-bo/master/docs/experiments/soc_math_mishra.png)  |  ![](https://raw.githubusercontent.com/thomas-young-2013/lite-bo/master/docs/experiments/soc_math_keane.png)

Multi-objective problems

DTLZ1-6-5             | ZDT2-3 
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/thomas-young-2013/lite-bo/master/docs/experiments/mo_math_dtlz1-6-5.png)  |  ![](https://raw.githubusercontent.com/thomas-young-2013/lite-bo/master/docs/experiments/mo_math_zdt2-3.png)

Multi-objective problems with constraints

CONSTR             | SRN 
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/thomas-young-2013/lite-bo/master/docs/experiments/moc_math_constr.png)  |  ![](https://raw.githubusercontent.com/thomas-young-2013/lite-bo/master/docs/experiments/moc_math_srn.png)

## Installation

**Installation via pip**

For Windows and Linux users, you can install by

```bash
pip install lite-bo
```

For macOS users, you need to install `pyrfr` correctly first, and then `pip install lite-bo`. 

The tips for installing `pyrfr` on macOS is [here](docs/source/installation/install-pyrfr-on-macos.md).

**Manual installation from the github source**

 ```bash
git clone https://github.com/thomas-young-2013/lite-bo.git && cd lite-bo
cat requirements.txt | xargs -n 1 -L 1 pip install
python setup.py install
 ```
macOS users still need to follow the [tips](docs/source/installation/install-pyrfr-on-macos.md) 
to install `pyrfr` correctly first.

## Quick Start

```python
import numpy as np
from litebo.utils.start_smbo import create_smbo


def branin(x):
    xs = x.get_dictionary()
    x1 = xs['x1']
    x2 = xs['x2']
    a = 1.
    b = 5.1 / (4. * np.pi ** 2)
    c = 5. / np.pi
    r = 6.
    s = 10.
    t = 1. / (8. * np.pi)
    ret = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return {'objs': (ret,)}


config_dict = {
    "optimizer": "SMBO",
    "parameters": {
        "x1": {
            "type": "float",
            "bound": [-5, 10],
            "default": 0
        },
        "x2": {
            "type": "float",
            "bound": [0, 15]
        },
    },
    "advisor_type": 'default',
    "max_runs": 90,
    "time_limit_per_trial": 5,
    "logging_dir": 'logs',
    "task_id": 'hp1'
}

bo = create_smbo(branin, **config_dict)
bo.run()
inc_value = bo.get_incumbent()
print('BO', '=' * 30)
print(inc_value)
```

## **Releases and Contributing**
OpenBox has a frequent release cycle. Please let us know if you encounter a bug by [filling an issue](https://github.com/thomas-young-2013/lite-bo/issues/new/choose).

We appreciate all contributions. If you are planning to contribute any bug-fixes, please do so without further discussions.

If you plan to contribute new features, new modules, etc. please first open an issue or reuse an existing issue, and discuss the feature with us.

To learn more about making a contribution to OpenBox, please refer to our [How-to contribution page](https://github.com/thomas-young-2013/lite-bo/blob/master/CONTRIBUTING.md). 

We appreciate all contributions and thank all the contributors!


## **Feedback**
* [File an issue](https://github.com/thomas-young-2013/lite-bo/issues) on GitHub.
* Email us via *liyang.cs@pku.edu.cn*.


## Related Projects

Targeting at openness and advancing AutoML ecosystems, we had also released few other open source projects.

* [VocalnoML](https://github.com/thomas-young-2013/soln-ml) : an open source system that provides end-to-end ML model training and inference capabilities.


## **License**

The entire codebase is under [MIT license](LICENSE)
