import os
import sys
import time
import argparse
sys.path.append(os.getcwd())
import litebo.core.distributed.nameserver as hpns
from litebo.optimizer.distributed_smbo import DistributedSMBO
from litebo.examples.worker_example import BraninWorker


parser = argparse.ArgumentParser(description='Cluster Parallel Experiments.')
parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=2)
parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=4)
parser.add_argument('--host', type=str, help='Hosts.', default='127.0.0.1')
parser.add_argument('--mode', type=str, choices=['worker', 'master'], default='master')

args = parser.parse_args()
host = args.host
run_id = 'cluster'

if args.mode == 'worker':
    time.sleep(5)
    w = BraninWorker(sleep_interval=0.5, run_id=run_id, host=host)
    w.run(background=False)
    exit(0)

# Start a nameserver:
# We now start the nameserver with the host name from above and a random open port (by setting the port to 0)
NS = hpns.NameServer(run_id=run_id)
ns_host, ns_port = NS.start()
# Most optimizers are so computationally inexpensive that we can afford to run a
# worker in parallel to it. Note that this one has to run in the background to
# not plock!
# w = BraninWorker(sleep_interval=0.5, run_id=run_id, host=host)
# w.run(background=True)

print(ns_host, ns_port)

# Run an optimizer
dsmbo = DistributedSMBO(task_id=run_id, config_space=BraninWorker.get_configspace())
dsmbo.run(min_n_workers=args.n_workers)

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
dsmbo.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
inc_value = dsmbo.get_incumbent()
print('BO', '='*30)
print(inc_value)
