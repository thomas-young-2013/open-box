import os
import sys
from litebo.utils import dependencies

__version__ = '0.5.4'
__author__ = 'ThomasYoung'

__MANDATORY_PACKAGES__ = """
numpy>=1.7.1
scipy>=0.18.1
ConfigSpace>=0.4.6,<0.5
scikit-learn>=0.18.0
pyrfr>=0.5.0
"""

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as fh:
    dependencies.verify_packages(fh.read())

supported_platforms = ['win32', 'linux2', 'linux1', 'darwin']
if sys.platform not in supported_platforms:
    raise ValueError('Lite-BO cannot run on platform-%s' % sys.platform)

if sys.version_info < (3, 5, 2):
    raise ValueError("Lite-BO requires Python 3.5.2 or newer.")

