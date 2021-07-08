# Installation Guide

## 1 System Requirements

Installation Requirements:
+ Python >= 3.6 (3.7 is recommended!)

Supported Systems:
+ Linux (Ubuntu, ...)
+ macOS
+ Windows

## 2 Preparations before Installation

We **STRONGLY** suggest you to create a Python environment via [Anaconda](https://www.anaconda.com/products/individual#Downloads):
```bash
conda create -n openbox3.7 python=3.7
conda activate openbox3.7
```

Then we recommend you to update your `pip` and `setuptools` as follows:
```bash
pip install pip setuptools --upgrade --user
```

## 3 Install OpenBox

### 3.1 Installation from PyPI

To install OpenBox from PyPI, simply run the following command:

```bash
pip install openbox
```

### 3.2 Manual Installation from Source

To install OpenBox using the source code, please run the following commands:

For Python >= 3.7:
```bash
git clone https://github.com/thomas-young-2013/open-box.git && cd open-box
cat requirements/main.txt | xargs -n 1 -L 1 pip install
python setup.py install
```

For Python == 3.6:
```bash
git clone https://github.com/thomas-young-2013/open-box.git && cd open-box
cat requirements/main_py36.txt | xargs -n 1 -L 1 pip install
python setup.py install
```

### 3.3 Test for Installation

You can run this quick start example to test your installation:

```python
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
```

If you encountered any problem during installation, please refer to the **Trouble Shooting** section.

## 4 Installation for Advanced Usage

To use advanced features such as `pyrfr` (probabilistic random forest) surrogate and get hyper-parameter 
importance from history, please refer to [Pyrfr Installation Guide](./install_pyrfr.md) to install `pyrfr`.

## 5 Trouble Shooting

If you encounter problems not listed below, please [File an issue](https://github.com/thomas-young-2013/open-box/issues) 
on GitHub or email us via *liyang.cs@pku.edu.cn*.

### Windows

+ 'Error: \[WinError 5\] Access denied'. Please open the command prompt with administrative privileges or 
append `--user` to the command line.

+ 'ERROR: Failed building wheel for ConfigSpace'. Please refer to [tips](./install_configspace_on_win_fix_vc.md).

+ For Windows users who have trouble installing lazy_import, please refer to [tips](./install-lazy_import-on-windows.md). (Deprecated in 0.7.10)

### macOS

+ For macOS users who have trouble installing pyrfr, please refer to [tips](./install-pyrfr-on-macos.md).

+ For macOS users who have trouble building scikit-learn, this [documentation](./openmp_macos.md) might help. 
