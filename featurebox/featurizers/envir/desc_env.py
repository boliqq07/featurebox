"""Use descriptors form ``pyXtal_FF``, in :mod:`featurebox.featurizers.descriptors`
"""
from featurebox.featurizers.descriptors.ACSF import ACSF
from featurebox.featurizers.descriptors.EAD import EAD
from featurebox.featurizers.descriptors.EAMD import EAMD
from featurebox.featurizers.descriptors.SO3 import SO3
from featurebox.featurizers.descriptors.SO4 import SO4_Bispectrum
from featurebox.featurizers.descriptors.SOAP import SOAP
from featurebox.featurizers.descriptors.behlerparrinello import BehlerParrinello
from featurebox.featurizers.descriptors.wACSF import wACSF
from featurebox.utils.look_json import mark_classes

DesDict = mark_classes([ACSF,
                        BehlerParrinello,
                        EAD,
                        EAMD,
                        SOAP,
                        SO3,
                        SO4_Bispectrum,
                        wACSF,
                        ])

for i, j in DesDict.items():
    locals()[i] = j
