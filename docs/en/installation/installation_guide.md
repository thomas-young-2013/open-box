# Installation Guide

## System Requirements

Installation Requirements:
+ Python >= 3.6
+ SWIG >= 3.0.12

## 1 Install SWIG

To install SWIG, please refer to [SWIG Installation Guide](./install_swig.md)

Make sure to install SWIG correctly before you install OpenBox.

## 2 Install OpenBox

### 2.1 Installation from PyPI

To install OpenBox from PyPI:

```bash
pip install openbox
```

### 2.2 Manual Installation from Source

To install OpenBox from command line, please type the following commands on the command line:

```bash
git clone https://github.com/thomas-young-2013/open-box.git && cd open-box
cat requirements/main.txt | xargs -n 1 -L 1 pip install
python setup.py install
```

## 3 Trouble Shooting

For macOS users who have trouble installing pyrfr, see the [tips](./install-pyrfr-on-macos.md).

For Windows users who have trouble installing lazy_import, see the [tips](./install-lazy_import-on-windows.md).
