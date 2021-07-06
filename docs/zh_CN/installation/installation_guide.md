# 安装指南

## 1 系统要求

安装要求：
+ Python >= 3.6

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

## 3 进阶功能安装

如果您想使用更高级的功能，比如使用 `pyrfr` （概率随机森林）作为代理模型，或根据历史计算超参数重要性，
请参考 [Pyrfr安装教程](./install_pyrfr.md) 并安装 `pyrfr`。

## 4 问题解答

对于 macOS 用户，如果您在安装 pyrfr 时遇到了困难，请参考 [提示](./install-pyrfr-on-macos.md)。

对于 macOS 用户，如果您在编译 scikit-learn 时遇到了困难。 [该教程](./openmp_macos.md) 或许对你有帮助。

对于 Windows 用户，如果您在安装 lazy_import 时遇到了困难，请参考 [提示](./install-lazy_import-on-windows.md)。
