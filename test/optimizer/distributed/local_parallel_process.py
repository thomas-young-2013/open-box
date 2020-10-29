import logging
logging.basicConfig(level=logging.INFO)

import argparse
import litebo.core.distributed.nameserver as hpns
from litebo.optimizer.distributed_smbo import DistributedSMBO
from litebo.examples.worker_example import BraninWorker

parser = argparse.ArgumentParser(description='Local and Parallel Execution.')
parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=4)
parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=2)
parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
args = parser.parse_args()


task_id = 'local_parallel_example'

if args.worker:
    w = BraninWorker(sleep_interval=0.5, nameserver='127.0.0.1', run_id=task_id)
    w.run(background=False)
    exit(0)

# Start a nameserver.
NS = hpns.NameServer(run_id='example3', host='127.0.0.1', port=9008)
NS.start()

dsmbo = DistributedSMBO(task_id=task_id, config_space=BraninWorker.get_configspace())

dsmbo.run()

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
dsmbo.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
inc_value = dsmbo.get_incumbent()
print('BO', '='*30)
print(inc_value)
