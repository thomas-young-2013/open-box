# SWIG Installation Guide

This tutorial helps you install SWIG 3.0.

## Linux

```bash
apt-get install swig3.0
ln -s /usr/bin/swig3.0 /usr/bin/swig
```

## MacOS

First, download SWIG from [SWIG Download Page](http://www.swig.org/download.html).
Clicking on **All releases** to see older releases. Then select the **'swig'** folder, 
not the 'macswig' folder. Version 3.0.12 is recommended.

Unzip the files after downloading.

Second, download pcre from [here](http://www.pcre.org).
Download 'pcre-8.44.tar.bz2' and rename it to 'pcre-8.44.tar'.
Then put it into the SWIG folder.

Third, run the commands:

```bash
./Tools/pcre-build.sh
./autogen.sh
./configure
make
sudo make install
```

## Windows

First, download SWIG for Windows from [SWIG Download Page](http://www.swig.org/download.html).
Clicking on **All releases** to see older releases. Then select the **'swigwin'** folder.
Version 3.0.12 is recommended.

Unzip the files after downloading.

Second, add SWIG folder to SYSTEM PATH.
