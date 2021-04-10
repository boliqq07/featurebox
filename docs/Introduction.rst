Introduction
==================

.. image:: img.jpg

Featurebox is one tool for material data **Generation** and **Selection**.

The main feature tools are:

============= =========================================
 Name         Application
------------- -----------------------------------------
 namespilt    Divide compound names to elemental proportion table.
 mp_access    Getting data from pymatgen conveniently.
 mapper       Getting each element data of compound.
 statistics   Getting each compound data with by combining element data with various arithmetical operation.
 union        Combined table data, such as polynomial extension.
 bond         Get bond data.
 graph        Integrated structure data tool.
 generator    Torch-like dataloader for data data in a non-uniform format.
============= =========================================

All the feature tools with  ``convert`` method for single case.
and ``fit_transform`` methods for case list.

The main binding selection tools are:

============= =========================================
 Name         Application
------------- -----------------------------------------
 backforward  Backforward selection
 corr         Correlation selection.
 exhaustion   Exhaustion selection.
 ga           Genetic algorithm selection.
============= =========================================

All the selection tools are ``sklearn-like``, with ``fit``, ``fit_transform`` methods .etc.

Note
::

    Where the binding means treat the binding features as one feature.
    And the binding features are selected or deleted synchronously.
