import sys
from litebo.utils import dependencies

__version__ = '0.0.1'
__author__ = 'UNNAMED'

__MANDATORY_PACKAGES__ = """
numpy>=1.7.1
scipy>=0.18.1
ConfigSpace>=0.4.6,<0.5
scikit-learn>=0.18.0
pyrfr>=0.5.0
"""
dependencies.verify_packages(__MANDATORY_PACKAGES__)

if sys.version_info < (3, 5, 2):
    raise ValueError("Lite-BO requires Python 3.5.2 or newer.")
