# 安装指南

## 系统要求

安装要求：
+ Python >= 3.6
+ SWIG >= 3.0.12

## 1 安装 SWIG

请参考我们的 [SWIG 安装指南](./install_swig.md) 来先安装 SWIG。

在安装OpenBox前，请保证SWIG已经被成功安装。

## 2 安装 OpenBox

### 2.1 使用 PyPI 安装

只需运行以下命令：

```bash
pip install openbox
```

### 2.2 从源代码手动安装

运行以下命令：

```bash
git clone https://github.com/thomas-young-2013/open-box.git && cd open-box
cat requirements/main.txt | xargs -n 1 -L 1 pip install
python setup.py install
```

## 3 问题解答

对于 macOS 用户，如果您在安装 pyrfr 时遇到了困难，请参考 [tips](./install-pyrfr-on-macos.md)。

对于 macOS 用户，如果您在编译 scikit-learn 时遇到了困难。 [这个](./openmp_macos.md) 或许对你有帮助。

对于 Windows 用户，如果您在安装 lazy_import 时遇到了困难，请参考 [tips](./install-lazy_import-on-windows.md)。
