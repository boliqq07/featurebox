# -*- coding: utf-8 -*-

# @Time    : 2019/10/18 14:27
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

import copy
import os
import re
import warnings
from itertools import chain
from typing import List

import pandas as pd

warnings.warn("This namesplit part is would be delete in future. "
              "please using: \nfrom featurebox.data.name_split import NameSplit", DeprecationWarning)

from .name_split import NameSplit