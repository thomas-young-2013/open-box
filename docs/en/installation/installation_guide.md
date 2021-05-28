# Installation Guide

## System Requirements

Installation Requirements:
+ Python >= 3.6
+ SWIG >= 3.0.12

## 1 Install SWIG

To install SWIG, please refer to [SWIG Installation Guide](./install_swig.md)

Make sure that SWIG is installed correctly installing OpenBox.

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

## 3 Trouble Shooting

For macOS users who have trouble installing pyrfr, refer to [tips](./install-pyrfr-on-macos.md).

For macOS users who have trouble building scikit-learn， this [documentation](./openmp_macos.md) might help. 

For Windows users who have trouble installing lazy_import, refer to [tips](./install-lazy_import-on-windows.md).
