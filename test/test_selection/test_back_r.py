from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, SVC

from featurebox.selection.backforward import BackForward
from featurebox.selection.ga import GA

# X, y = load_boston(return_X_y=True)
# svr = SVR()
# bf = BackForward(svr, primary_feature=4, random_state=1, refit=True)
# new_x = bf.fit_transform(X[:50], y[:50])
# test_score = bf.score(X[50:], y[50:])


param_grid1 = {'C': [50, 10, 5, 2.5, 1, 0.1]}
X, y = load_iris(return_X_y=True)
svc = SVC()
gd = GridSearchCV(svc, cv=5, param_grid=param_grid1)

bf = GA(estimator=gd, random_state=3)
bf.fit(X[:, :6], y)
