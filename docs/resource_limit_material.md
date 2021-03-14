# Limit runtime.
Python package `timeout_decorator`: https://github.com/pnpnpn/timeout-decorator.

Support main thread and multiple processes.
Additional resources can be found:
1. https://codereview.stackexchange.com/questions/142828/python-executer-that-kills-processes-after-a-timeout


## Limit memory usage.
Check file in `utils.decorators`, and this implementation ONLY supports Linux-like systems.

RECOMMEND the users to use package `psutil`: https://github.com/giampaolo/psutil.

Useful materials: 
1. https://docs.python.org/3.4/library/resource.html
2. https://github.com/giampaolo/psutil
