# Overview

OpenBox is a generalized and efficient system designed for blackbox optimization (BBO).
Its design satisfies the following desiderata:

+ **Ease of use**: Minimal user effort, and user-friendly visualization for tracking and managing BBO tasks.

+ **Consistent performance**: Host state-of-the-art optimization algorithms; choose the proper algorithm automatically.

+ **Resource-aware management**: Give cost-model-based advice to users, e.g., minimal workers or time-budget.

+ **Scalability**: Scale to dimensions on the number of input variables, objectives, tasks, trials, and parallel evaluations.

+ **High efficiency**: Effective use of parallel resources, system optimization with transfer-learning and multi-fidelities, etc.

+ **Fault tolerance**, **extensibility**, and **data privacy protection**.


The figure below shows high-level architecture of OpenBox service.

<p align="center">
<img src="https://raw.githubusercontent.com/thomas-young-2013/open-box/master/docs/imgs/sys_framework.svg" width="90%">
</p>

## Main Components

+ **Service Master** is responsible for node management, load balance, and fault tolerance.

+ **Task Database** holds the states of all tasks. 

+ **Suggestion Service** creates new configurations for each task.

+ **REST API** establishes the bridge between users/workers and suggestion service. 

+ **Evaluation workers** are provided and owned by the users.


## Deployment Artifacts

### Standalone Python package
You can install the released package and use it using Python.

### Distributed BBO service
We adopt the "BBO as a service" paradigm and implement OpenBox as a managed general service for black-box optimization.
Users can access this service via REST API conveniently, and do not need to worry about other issues such as
environment setup, software maintenance, programming, and optimization of the execution. Moreover, we also provide a
Web UI, through which users can easily track and manage the tasks.


## Scenarios

The goal of black-box optimization is to find a configuration that approaches the global optimum 
as rapidly as possible since evaluation of objective functions is often expensive.
Traditional BBO with a single objective has many applications:

+ Automatic A/B testing.
+ Experimental design.
+ Database knob tuning.
+ Automatic hyper-parameter tuning, one of the most indispensable components in AutoML systems,
  where the task is to minimize the validation error of a machine learning algorithm as a function of its
  hyper-parameters.

Recently, generalized BBO emerges and has been applied to many areas:

+ Processor architecture and circuit design.
+ Resource allocation.
+ Automatic chemical design.

Generalized BBO requires more general functionalities that may not be supported by traditional BBO,
such as multiple objectives and constraints.


## Performance Rank

<p align="center">
<img src="https://raw.githubusercontent.com/thomas-young-2013/open-box/master/docs/imgs/ranking_lgb_7.svg" width="80%">
</p>

Performance rank of AutoML Benchmark on LightGBM on 25 datasets. The lower is the better.

