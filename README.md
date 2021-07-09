<p align="center">
<img src="https://raw.githubusercontent.com/thomas-young-2013/open-box/master/docs/imgs/logo.png" width="68%">
</p>

-----------

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/thomas-young-2013/open-box/blob/master/LICENSE)
[![Build Status](https://api.travis-ci.org/thomas-young-2013/open-box.svg?branch=master)](https://api.travis-ci.org/thomas-young-2013)
[![Issues](https://img.shields.io/github/issues-raw/thomas-young-2013/open-box.svg)](https://github.com/thomas-young-2013/open-box/issues?q=is%3Aissue+is%3Aopen)
[![Bugs](https://img.shields.io/github/issues/thomas-young-2013/open-box/bug.svg)](https://github.com/thomas-young-2013/open-box/issues?q=is%3Aissue+is%3Aopen+label%3Abug)
[![Pull Requests](https://img.shields.io/github/issues-pr-raw/thomas-young-2013/open-box.svg)](https://github.com/thomas-young-2013/open-box/pulls?q=is%3Apr+is%3Aopen)
[![Version](https://img.shields.io/github/release/thomas-young-2013/open-box.svg)](https://github.com/thomas-young-2013/open-box/releases)
[![Join the chat at https://gitter.im/bbo-open-box](https://badges.gitter.im/bbo-open-box.svg)](https://gitter.im/bbo-open-box?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Documentation Status](https://readthedocs.org/projects/open-box/badge/?version=latest)](https://open-box.readthedocs.io/en/latest/?badge=latest)

[OpenBox Doc](https://open-box.readthedocs.io) | [简体中文](README_zh_CN.md)

## OpenBox: Generalized and Efficient Blackbox Optimization System.
OpenBox is an efficient and generalized blackbox optimization (BBO) system, which owns the following characteristics:
1. BBO with multiple objectives and constraints.
2. BBO with transfer learning.
3. BBO with distributed parallelization.
4. BBO with multi-fidelity acceleration.
5. BBO with early stops.


## Deployment Artifacts
#### Standalone Python package.
Users can install the released package and use it using Python.

#### Distributed BBO service.
We adopt the "BBO as a service" paradigm and implement OpenBox as a managed general service for black-box optimization. Users can access this service via REST API conveniently, and do not need to worry about other issues such as environment setup, software maintenance, programming, and optimization of the execution. Moreover, we also provide a Web UI,
through which users can easily track and manage the tasks.


## Design Goal

OpenBox’s design satisfies the following desiderata:
+ **Ease of use**: Minimal user effort, and user-friendly visualization for tracking and managing BBO tasks.

+ **Consistent performance**: Host state-of-the-art optimization algorithms; Choose the proper algorithm automatically.

+ **Resource-aware management**: Give cost-model-based advice to users, e.g., minimal workers or time-budget.

+ **Scalability**: Scale to dimensions on the number of input variables, objectives, tasks, trials, and parallel evaluations.

+ **High efficiency**: Effective use of parallel resources, system optimization with transfer-learning and multi-fidelities, etc.

+ **Fault tolerance**, **extensibility**, and **data privacy protection**.

## Links
+ [Documentations](https://open-box.readthedocs.io/en/latest/?badge=latest)
+ [Examples](https://github.com/thomas-young-2013/open-box/tree/master/examples)
+ [Pypi package](https://pypi.org/project/open-box/)
+ Conda package: [to appear soon]()
+ Blog post: [to appear soon]()

## Application Tutorials
+ [Tuning LightGBM with OpenBox](https://github.com/thomas-young-2013/open-box/blob/master/docs/en/articles/openbox_LightGBM.md)
+ [Tuning XGBoost using OpenBox](https://github.com/thomas-young-2013/open-box/blob/master/docs/en/articles/openbox_XGBoost.md)

## Benchmark Results

Single-objective problems
Ackley-4                  | Hartmann
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/thomas-young-2013/open-box/master/docs/experiments/so_math_ackley-4.png)  |  ![](https://raw.githubusercontent.com/thomas-young-2013/open-box/master/docs/experiments/so_math_hartmann.png)

Single-objective problems with constraints
Mishra                  | Keane-10
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/thomas-young-2013/open-box/master/docs/experiments/soc_math_mishra.png)  |  ![](https://raw.githubusercontent.com/thomas-young-2013/open-box/master/docs/experiments/soc_math_keane.png)

Multi-objective problems

DTLZ1-6-5             | ZDT2-3 
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/thomas-young-2013/open-box/master/docs/experiments/mo_math_dtlz1-6-5.png)  |  ![](https://raw.githubusercontent.com/thomas-young-2013/open-box/master/docs/experiments/mo_math_zdt2-3.png)

Multi-objective problems with constraints

CONSTR             | SRN 
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/thomas-young-2013/open-box/master/docs/experiments/moc_math_constr.png)  |  ![](https://raw.githubusercontent.com/thomas-young-2013/open-box/master/docs/experiments/moc_math_srn.png)

## Installation

### System Requirements

Installation Requirements:
+ Python >= 3.6 (3.7 is recommended!)

We **STRONGLY** suggest you to create a Python environment via [Anaconda](https://www.anaconda.com/products/individual#Downloads):
```bash
conda create -n openbox3.7 python=3.7
conda activate openbox3.7
```

Then we recommend you to update your `pip` and `setuptools` as follows:
```bash
pip install pip setuptools --upgrade --user
```

### Installation from PyPI

To install OpenBox from PyPI:

```bash
pip install openbox --user
```

### Manual Installation from Source

To install the newest version of OpenBox, please type the following scripts on the command line:

```bash
git clone https://github.com/thomas-young-2013/open-box.git && cd open-box
cat requirements/main.txt | xargs -n 1 -L 1 pip install
python setup.py install --user --prefix=
```

For more detailed installation instructions, please refer to our [Installation Guide Document](https://open-box.readthedocs.io/en/latest/installation/installation_guide.html).

## Quick Start

```python
import numpy as np
from openbox import Optimizer, sp

# Define Search Space
space = sp.Space()
x1 = sp.Real("x1", -5, 10, default_value=0)
x2 = sp.Real("x2", 0, 15, default_value=0)
space.add_variables([x1, x2])

# Define Objective Function
def branin(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2-5.1/(4*np.pi**2)*x1**2+5/np.pi*x1-6)**2+10*(1-1/(8*np.pi))*np.cos(x1)+10
    return y

# Run
if __name__ == '__main__':
    opt = Optimizer(branin, space, max_runs=50, task_id='quick_start')
    history = opt.run()
    print(history)
```

## **Releases and Contributing**
OpenBox has a frequent release cycle. Please let us know if you encounter a bug by [filling an issue](https://github.com/thomas-young-2013/open-box/issues/new/choose).

We appreciate all contributions. If you are planning to contribute any bug-fixes, please do so without further discussions.

If you plan to contribute new features, new modules, etc. please first open an issue or reuse an existing issue, and discuss the feature with us.

To learn more about making a contribution to OpenBox, please refer to our [How-to contribution page](https://github.com/thomas-young-2013/open-box/blob/master/CONTRIBUTING.md). 

We appreciate all contributions and thank all the contributors!


## **Feedback**
* [File an issue](https://github.com/thomas-young-2013/open-box/issues) on GitHub.
* Email us via *liyang.cs@pku.edu.cn*.


## Related Projects

Targeting at openness and advancing AutoML ecosystems, we had also released few other open source projects.

* [VolcanoML](https://github.com/thomas-young-2013/soln-ml) : an open source system that provides end-to-end ML model training and inference capabilities.


---------------------
## **Related Publications**

**OpenBox: A Generalized Black-box Optimization Service**
Yang Li, Yu Shen, Wentao Zhang, Yuanwei Chen, Huaijun Jiang, Mingchao Liu, Jiawei Jiang, Jinyang Gao, Wentao Wu, Zhi Yang, Ce Zhang, Bin Cui; ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2021).

**MFES-HB: Efficient Hyperband with Multi-Fidelity Quality Measurements**
Yang Li, Yu Shen, Jiawei Jiang, Jinyang Gao, Ce Zhang, Bin Cui; The Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI 2021).

## **License**

The entire codebase is under [MIT license](LICENSE).
