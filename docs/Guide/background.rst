Sample Data and Background
===========================

Sample data
::::::::::::

Download: `structure_list <https://github.com/boliqq07/featurebox/blob/master/test/structure_data/sample_data.pkl_pd>`_

Usage:

    >>> import pandas as pd
    >>> from pymatgen.core import Structure
    >>> structures = structure_list = pd.read_pickle("sample_data.pkl_pd")
    >>> structure = structurei = structure_list[0]


Background
::::::::::::

The Structure from ``pymatgen`` is one class to represent the crystal structure data, which contain all message
of atoms and their sites.

link:

`pymatgen <https://pymatgen.org/>`_

`pymatgen Structure <https://pymatgen.org/usage.html#reading-and-writing-structures-molecules>`_

From this type data, we could extract necessary message.

such as for batch data (used by ``transform`` ):

  >>> name_data = [[{str(i.symbol): 1} for i in si.species] for si in structure_list]
  >>> number_data = [[i.specie.Z for i in si] for si in structure_list]

such as for single case (used by ``convert`` ):

  >>> structure_1 = structure_list[0]
  >>> name_single = [{str(i.symbol): 1} for i in structure_1.species]
  >>> number_single = [i.specie.Z for i in structure_1]


In this packages, we accept data with type like ``name_data`` , ``number_data``  as input data.

Meanwhile, The above extract are built in code, thus we could accept ``structure_1`` ,
``structure_list`` directly.

The ``ase.Atoms`` could convert by Adaptor ``AseAtomsAdaptor`` from pymatgen or ``featurebox.utils.general.AAA`` .

Of course, The data ``name`` data , ``number`` data could build by yourself from you code.