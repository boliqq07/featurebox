"""
Check the element.
"""
from typing import List, Callable, Text
import numpy as np

ALL_ELE_NAME = ("H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
                "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
                "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo",
                "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba",
                "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
                "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
                "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
                "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
                "Nh", "Fl", "Mc", "Lv", "Ts", "Og")  # 118

ALL_N_ELE_MAP = {v + 1: k for k, v in zip(ALL_ELE_NAME, range(len(ALL_ELE_NAME)))}
ALL_ELE_N_MAP = {k: v + 1 for k, v in zip(ALL_ELE_NAME, range(len(ALL_ELE_NAME)))}

AVAILABLE_ELE_NAME = ('H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S',
                      'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
                      'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
                      'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                      'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os',
                      'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
                      # ######### 'Po', 'At', 'Rn', 'Fr', 'Ra', # 84-88
                      'Ac', 'Th', 'Pa', 'U',
                      # ########## "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg",
                      # ########## "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
                      )

AVAILABLE_ELE_NUMBER = tuple(list((range(1, 84))) + [89, 90, 91, 92])


class CheckElements:
    """Check the element in preprocessing."""

    def __init__(self, check_method: str = "name", func: Callable = lambda x: x):
        """

        Parameters
        ----------
        check_method: str
            Check by number or name of element.
        func: callable
            Processing for elements.
            such as for element in pymatgen:
                func=lambda x: [x.Z, ]
                func=lambda x: [x.name, ]

        Examples
        ---------
        >>> ce = CheckElements.from_list(check_method="name",grouped=False)
        >>> ce.check(["Na","Al","Ta"])
        ['Na', 'Al', 'Ta']
        >>> ce = CheckElements.from_list(check_method="name",grouped=True)
        >>> ce.check([["Na","Al"],["Na","Ta"]])
        [['Na', 'Al'], ['Na', 'Ta']]
        >>> ce.check([["Na","Al"],["Na","Ra"],["Zn","H"]])
        The 1 (st,ed,th) sample ['Na', 'Ra'] is with element out of AVAILABLE_ELE_NAME
         please to check_data.py for more information.
        [['Na', 'Al'], ['Zn', 'H']]
        >>> ce.passed_idx()
        array([0, 2], dtype=int64)

        Examples
        ---------
        >>> ce = CheckElements.from_pymatgen_structures()
        ...

        """

        if check_method == "name":
            check_method = AVAILABLE_ELE_NAME
        elif check_method == "number":
            check_method = AVAILABLE_ELE_NUMBER
        else:
            raise TypeError("check_method='name' or 'name'")
        self.check_method = check_method
        self.func = func
        self.mark = []

    def check(self, samples: List) -> List:
        """

        Parameters
        ----------
        samples: list
            names or numbers, or pymatgen.structures

        Returns
        -------
        result: list
            list of filtered structure.
        """

        self.mark = []
        structures_t = []
        for i, si in enumerate(samples):
            si_ = self.func(si)
            if np.all([True if ei in self.check_method else False for ei in si_]):
                structures_t.append(si)
                self.mark.append(1)
            else:
                print("The {} (st,ed,th) sample {} is with element out of AVAILABLE_ELE_NAME\n".format(i, str(si)),
                      "please to check_data.py for more information.")
                self.mark.append(0)
        return structures_t

    def passed_idx(self) -> np.ndarray:
        """The mark for all structures, return np.ndarray index"""
        return np.where(np.array(self.mark) == 1)[0]

    @classmethod
    def from_list(cls, check_method="name", grouped="False"):
        """Get checker for list"""
        if grouped:
            func = lambda x: x
        else:
            func = lambda x: [x, ]
        return cls(check_method=check_method, func=func)

    @classmethod
    def from_pymatgen_structures(cls):
        """Get checker for pymatgen.Structure"""
        func = lambda x: x.atomic_numbers
        return cls(check_method="number", func=func)
