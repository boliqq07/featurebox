from featurebox.featurizers.mapper import AtomJsonMap, AtomTableMap

from sklearn.datasets import load_boston
from featurebox.selection.corr import Corr

x, y = load_boston(return_X_y=True)
co = Corr(threshold=0.5, muti_index=[0, 8], muti_grade=2)
newx = co.fit_transform(x)
print(x.shape)
print(newx.shape)


