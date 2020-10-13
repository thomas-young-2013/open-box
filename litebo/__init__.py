import os
import sys
from litebo.utils import dependencies

__version__ = '0.5.9'
__author__ = 'Yang Li'

__MANDATORY_PACKAGES__ = """
cython
pyrfr>=0.5.0
setuptools
numpy>=1.7.1
scipy>=0.18.1
ConfigSpace>=0.4.6,<0.5
scikit-learn==0.21.3
"""

# with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as fh:
#     dependencies.verify_packages(fh.read())
dependencies.verify_packages(__MANDATORY_PACKAGES__)

supported_platforms = ['win32', 'linux2', 'linux1', 'linux', 'darwin']
if sys.platform not in supported_platforms:
    raise ValueError('Lite-BO cannot run on platform-%s' % sys.platform)

if sys.version_info < (3, 5, 2):
    raise ValueError("Lite-BO requires Python 3.5.2 or newer.")

