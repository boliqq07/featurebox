from sklearn.datasets import load_boston
from sklearn.svm import SVR

from featurebox.selection.backforward import BackForward

X, y = load_boston(return_X_y=True)
svr = SVR()
bf = BackForward(svr, primary_feature=4, random_state=1, refit=True)
new_x = bf.fit_transform(X[:50], y[:50])
test_score = bf.score(X[50:], y[50:])