# Installation Guide

## 1 System Requirements

Installation Requirements:
+ Python >= 3.6

## 2 Install OpenBox

### 2.1 Installation from PyPI

To install OpenBox from PyPI, simply run the following command:

```bash
pip install openbox
```

### 2.2 Manual Installation from Source

To install OpenBox using the source code, please run the following commands:

```bash
git clone https://github.com/thomas-young-2013/open-box.git && cd open-box
cat requirements/main.txt | xargs -n 1 -L 1 pip install
python setup.py install
```

## 3 Installation for Advanced Usage

To use advanced features such as `pyrfr` (probabilistic random forest) surrogate and get hyper-parameter 
importance from history, please refer to [Pyrfr Installation Guide](./install_pyrfr.md) to install `pyrfr`.

## 4 Trouble Shooting

For macOS users who have trouble installing pyrfr, refer to [tips](./install-pyrfr-on-macos.md).

For macOS users who have trouble building scikit-learn, this [documentation](./openmp_macos.md) might help. 

For Windows users who have trouble installing lazy_import, refer to [tips](./install-lazy_import-on-windows.md).
