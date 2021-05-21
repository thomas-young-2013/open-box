# Installation Guide

## System Requirements

Installation Requirements:
+ Python >= 3.6
+ SWIG >= 3.0.12

Make sure to install SWIG correctly before you install OpenBox.

To install SWIG, please refer to [SWIG Installation Guide](./install_swig.md)

## Installation from PyPI

To install OpenBox from PyPI:

```bash
pip install openbox
```

## Manual Installation from Source

To install OpenBox from command line, please type the following commands on the command line:

```bash
git clone https://github.com/thomas-young-2013/open-box.git && cd open-box
cat requirements/main.txt | xargs -n 1 -L 1 pip install
python setup.py install
```

## Trouble Shooting

For macOS users who have trouble installing pyrfr, see the [tips](./install-pyrfr-on-macos.md).

For Windows users who have trouble installing lazy_import, see the [tips](./install-lazy_import-on-windows.md).
