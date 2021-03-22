# Open-BOX: Generalized and Efficient Blackbox Optimization System.
Open-BOX is an efficient and generalized blackbox optimization (BBO) system, which owns the following characteristics:
1. Basic BBO algorithms.
2. BBO with constraints.
3. BBO with multiple objectives.
4. BBO with transfer learning.
5. BBO with distributed parallelization.
6. BBO with multi-fidelity acceleration.
7. BBO with early stops.


## Deployment Artifacts
### Standalone Python package.
Users can install the released package and use it using Python.

### Distributed BBO service.
We adopt the "BBO as a service" paradigm and implement OpenBox as a managed general service for black-box optimization. Users can access this service via REST API conveniently, and do not need to worry about other issues such as environment setup, software maintenance, programming, and optimization of the execution. Moreover, we also provide a Web UI,
through which users can easily track and manage the tasks.
