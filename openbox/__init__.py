# This en code is licensed under the MIT license found in the
# LICENSE file in the root directory of this en tree.

from .pkginfo import version as __version__, package_name

from .utils import space
from .utils import space as sp

from .optimizer.generic_smbo import SMBO as Optimizer
from .optimizer.parallel_smbo import pSMBO as ParallelOptimizer
from .optimizer.message_queue_smbo import mqSMBO as DistributedOptimizer
from .core.message_queue.worker import Worker as DistributedWorker
from .optimizer.nsga_optimizer import NSGAOptimizer

from .utils.start_smbo import create_smbo as create_optimizer

from .utils.tuning import get_config_space, get_objective_function

from .utils.test_install import run_test

__all__ = [
    "__version__", "__package__", "package_name",
    "sp", "space",
    "Optimizer", "ParallelOptimizer", "DistributedOptimizer", "DistributedWorker",
    "NSGAOptimizer",
    "create_optimizer",
    "get_config_space", "get_objective_function",
    "run_test",
]
