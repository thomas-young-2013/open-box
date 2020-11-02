import os
import sys
import logging
import argparse
sys.path.append(os.getcwd())
import litebo.core.distributed.nameserver as hpns
from litebo.optimizer.distributed_smbo import DistributedSMBO
from litebo.examples.worker_example import BraninWorker

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description='Local and Parallel Execution.')
parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=4)
parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=2)
parser.add_argument('--mode', type=str, choices=['worker', 'master'], default='master')
args = parser.parse_args()


task_id = 'local_parallel_example'

if args.mode == 'worker':
    w = BraninWorker(sleep_interval=0.5, run_id=task_id)
    w.run(background=False)
    exit(0)

# Start a nameserver.
NS = hpns.NameServer(run_id='example3')
NS.start()

dsmbo = DistributedSMBO(task_id=task_id, config_space=BraninWorker.get_configspace())

dsmbo.run(min_n_workers=args.n_workers)

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
dsmbo.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
inc_value = dsmbo.get_incumbent()
print('BO', '='*30)
print(inc_value)
