.. Open-BOX documentation master file, created by
   sphinx-quickstart on Fri Dec 23 13:31:47 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/thomas-young-2013/lite-bo

Open-BOX documentation
======================

Open-BOX is an efficient and generalized blackbox optimization (BBO) system.

Features described in this documentation are classified by release status:

  *Stable:*  These features will be maintained long-term and there should generally
  be no major performance limitations or gaps in documentation.
  We also expect to maintain backwards compatibility (although
  breaking changes can happen and notice will be given one release ahead
  of time).

  *Beta:*  Features are tagged as Beta because the API may change based on
  user feedback, because the performance needs to improve, or because
  coverage across operators is not yet complete. For Beta features, we are
  committing to seeing the feature through to the Stable classification.
  We are not, however, committing to backwards compatibility.

  *Prototype:*  These features are typically not available as part of
  binary distributions like PyPI or Conda, except sometimes behind run-time
  flags, and are at an early stage for feedback and testing.


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Introduction

   introduction/overview

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Installation

   installation/installation_guide

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Getting Started

   getting_started/quick_example

.. toctree::
   :maxdepth: 1
   :caption: Usage Examples

   usage_examples/single_objective_hpo
   usage_examples/single_objective_with_constraint
   usage_examples/multi_objective
   usage_examples/multi_objective_with_constraint

.. toctree::
   :maxdepth: 1
   :caption: Advanced Usage

   advanced_usage/parallel_on_local
   advanced_usage/distributed_service
   advanced_usage/transfer_learning

.. toctree::
   :maxdepth: 1
   :caption: Manual

   manual/manual

.. toctree::
   :maxdepth: 1
   :caption: Doc Examples (for test)

   examples/note1
   examples/pytorch_rst_complex_example

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
