## Error: command '/usr/bin/clang' failed with exit status 1
When you install the `pyrfr` package, you might get this problem.
CFLAGS=-stdlib=libc++ pip install smac

## Install the variant of SMAC3=0.8.0.
pip install git+https://git@github.com/thomas-young-2013/SMAC3.git@iter_smac

## Install the newest AutoSklearn.
pip install git+https://git@github.com/thomas-young-2013/auto-sklearn.git@dev

## Basic important requirements.
pyrfr==0.8.0
smac==0.8.0 (our implemented variant)
scikit-learn==0.22.1
auto-sklearn==0.6.0 (variant based on the newest version)
