import time
import sys
import traceback
from litebo.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from litebo.utils.limit import time_limit, TimeoutException
from litebo.core.message_queue.worker_messager import WorkerMessager


class Worker(object):
    def __init__(self, objective_function, ip="127.0.0.1", port=13579, authkey=b'abc'):
        self.objective_function = objective_function
        self.worker_messager = WorkerMessager(ip, port, authkey)

    def run(self):
        while True:
            # Get config
            try:
                msg = self.worker_messager.receive_message()
            except Exception as e:
                print("Worker receive message error:", str(e))
                return
            if msg is None:
                # Wait for configs
                time.sleep(1)
                continue
            print("Worker: get config. start working.")
            config, time_limit_per_trial = msg

            # Start working
            trial_state = SUCCESS
            try:
                args, kwargs = (config,), dict()
                timeout_status, _result = time_limit(self.objective_function,
                                                     time_limit_per_trial,
                                                     args=args, kwargs=kwargs)
                if timeout_status:
                    raise TimeoutException(
                        'Timeout: time limit for this evaluation is %.1fs' % time_limit_per_trial)
                else:
                    if _result is None:
                        objs = None
                        constraints = None
                    elif isinstance(_result, dict):  # recommended usage
                        objs = _result['objs']
                        constraints = _result.get('constraints', None)
                    elif isinstance(_result, (int, float)):
                        objs = (_result,)
                        constraints = None
                    else:
                        objs = _result
                        constraints = None
            except Exception as e:
                if isinstance(e, TimeoutException):
                    trial_state = TIMEOUT
                else:
                    traceback.print_exc(file=sys.stdout)
                    trial_state = FAILED
                objs = None
                constraints = None
            observation = [config, trial_state, constraints, objs]

            # Send result
            print("Worker: observation=%s. sending result." % str(observation))
            try:
                self.worker_messager.send_message(observation)
            except Exception as e:
                print("Worker send message error:", str(e))
                return
