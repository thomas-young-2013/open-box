.. OpenBox documentation master file, created by
   sphinx-quickstart on Fri Dec 23 13:31:47 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/thomas-young-2013/open-box

OpenBox: Generalized and Efficient Blackbox Optimization System
================================================================

**OpenBox** is an open-source system designed for black-box
optimization (BBO). OpenBox can solve the generalized BBO problems efficiently.
It supports 1) traditional BBO problems with single objective (e.g.,
hyper-parameter optimization), 2) multi-objective BBO problems,
3) BBO with constraints, and 4) multi-objective BBO problems with constraints.
Moreover, it provides additional optimizations with transfer learning, distributed parallel evaluation, multi-fidelity optimization,
etc.
OpenBox has two kinds for software artifacts: standalone python package and online
BBO service.

----------------------------------------

Who should consider using OpenBox
=================================

* Those who want to tune hyper-parameters for their ML tasks automatically.
* Those who want to find the optimal configuration for their configuration search tasks (e.g., knobs tuning in database).
* Data Platform owners who want to provide BBO service in their platform.
* Researchers and data scientists who want to easily solve the generalized BBO problems.


----------------------------------------

.. _openbox-characteristics--capabilities:

OpenBox characteristics & capabilities
======================================

1. Basic BBO algorithms.

2. BBO with constraints.

3. BBO with multiple objectives.

4. BBO with transfer learning.

5. BBO with distributed parallelization.

6. BBO with multi-fidelity acceleration.

7. BBO with early stops.

----------------------------------------

Installation
============

System Requirements
-------------------

Installation Requirements:

-  Python >= 3.6

-  SWIG >= 3.0.12

Make sure to install SWIG correctly before you install OpenBox.

To install SWIG, please refer to `SWIG Installation
Guide <https://github.com/thomas-young-2013/open-box/blob/master/docs/source/installation/install_swig.md>`__

Installation from PyPI
----------------------

To install OpenBox from PyPI:

.. code:: shell

    pip install openbox

Manual Installation from Source
-------------------------------

To install OpenBox from command line, please type the following commands
on the command line:

.. code:: shell

    git clone https://github.com/thomas-young-2013/open-box.git && cd open-box
    cat requirements/main.txt | xargs -n 1 -L 1 pip install
    python setup.py install

The tips for installing ``pyrfr`` on macOS is
`here <https://github.com/thomas-young-2013/open-box/blob/master/docs/source/installation/install-pyrfr-on-macos.md>`__.
Please make sure you installed ``pyrfr`` correctly.

The tips for installing ``lazy_import`` on Windows is
`here <https://github.com/thomas-young-2013/open-box/blob/master/docs/source/installation/install-lazy_import-on-windows.md>`__.

----------------------------------------

Quick Start
===========

.. code:: python

    import numpy as np
    from openbox.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter
    from openbox.optimizer.generic_smbo import SMBO

    # Define Configuration Space
    config_space = ConfigurationSpace()
    x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
    x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
    config_space.add_hyperparameters([x1, x2])

    # Define Objective Function
    def branin(config):
        x1, x2 = config['x1'], config['x2']
        y = (x2-5.1/(4*np.pi**2)*x1**2+5/np.pi*x1-6)**2+10*(1-1/(8*np.pi))*np.cos(x1)+10
        return y

    # Run
    if __name__ == '__main__':
        bo = SMBO(branin, config_space, max_runs=50, task_id='quick_start')
        history = bo.run()
        print(history)

----------------------------------------

Documentation
=============

-  To learn about what's OpenBox, read the `OpenBox
   Overview <./overview.html>`__.

-  To get yourself familiar with how to use OpenBox, read the
   `documentation <.>`__.

-  To get started and install OpenBox on your system, please refer to
   `Install OpenBox <./installation/installation_guide.html>`__.

Related Articles
================

-  `Tuning LightGBM with
   OpenBox <https://github.com/thomas-young-2013/open-box/blob/master/docs/en_US/tutorials/openbox_LightGBM.md>`__

-  `Tuning XGBoost using
   OpenBox <https://github.com/thomas-young-2013/open-box/blob/master/docs/en_US/tutorials/openbox_XGBoost.md>`__

----------------------------------------

Releases and Contributing
=========================

OpenBox has a frequent release cycle. Please let us know if you
encounter a bug by `filling an
issue <https://github.com/thomas-young-2013/open-box/issues/new/choose>`__.

We appreciate all contributions. If you are planning to contribute any
bug-fixes, please do so without further discussions.

If you plan to contribute new features, new modules, etc. please first
open an issue or reuse an existing issue, and discuss the feature with
us.

To learn more about making a contribution to OpenBox, please refer to
our `How-to contribution
page <https://github.com/thomas-young-2013/open-box/blob/master/CONTRIBUTING.md>`__.

We appreciate all contributions and thank all the contributors!

----------------------------------------

Related Publications
====================

| **OpenBox: A Generalized Black-box Optimization Service**
| Yang Li, Yu Shen, Wentao Zhang, Yuanwei Chen, Huaijun Jiang, Mingchao
  Liu, Jiawei Jiang, Jinyang Gao, Wentao Wu, Zhi Yang, Ce Zhang,
  Bin Cui; ACM SIGKDD Conference on Knowledge Discovery and Data Mining
  (2021).



Related Project
===============

Targeting at openness and advancing AutoML ecosystems, we had also
released few other open source projects.

-  `VocalnoML <https://github.com/thomas-young-2013/soln-ml>`__ : an
   open source system that provides end-to-end ML model training and
   inference capabilities.

External Repositories
=====================

(To be filled)

----------------------------------------

Feedback
========

-  `File an
   issue <https://github.com/thomas-young-2013/open-box/issues>`__ on
   GitHub.

-  Email us via liyang.cs@pku.edu.cn.

License
=======

The entire codebase is under `MIT license <LICENSE>`__.


----------------------------------------

Contents
--------

..  toctree::
    :caption: Table of Contents
    :maxdepth: 2
    :titlesonly:

    Overview <overview>
    Installation <installation/installation_guide>
    Quick Start <quick_start/quick_start>
    Examples <examples/examples>
    Advanced Usage <advanced_usage/advanced_usage>
    OpenBox as Service <openbox_as_service/openbox_as_service>
    Research and Publications <research_and_publications/research_and_publications>
    Change Logs <change_logs/change_logs>


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
