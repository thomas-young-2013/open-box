# Use Case

The goal of black-box optimization is to find a configuration that
approaches the global optimum as rapidly as possible since evaluation of objective functions is often expensive.

Traditional BBO with a single objective has many applications:
1) automatic A/B testing.
2) experimental design.
3) knobs tuning in database.
4) automatic hyper-parameter tuning, one of the most indispensable components in AutoML systems,
where the task is to minimize the validation error of a machine learning algorithm as a function of its
hyper-parameters. 

Recently, generalized BBO emerges and has been applied to many areas:
1) processor architecture and circuit design.
2) resource allocation.
3) automatic chemical design.

Generalized BBO requires more general functionalities that may not be supported by traditional BBO,
such as multiple objectives and constraints.

