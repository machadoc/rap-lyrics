from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

class MLR:
    def __init__(self):
        self.mlr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

    def cross_validation(self, X, y):
        hyperparameter = {
            'logisticregression__C': [0.01, 0.1, 1, 10, 100],
            'logisticregression__solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
            'logisticregression__max_iter': [10000],
            'logisticregression__tol': [1e-3]
        }

        pipeline = make_pipeline(StandardScaler(), LogisticRegression())

        grid_search = GridSearchCV(pipeline, hyperparameter, cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(X, y)

        return grid_search.best_params_

    def fit(self, X, y, hyperparameters):
        solver = hyperparameters.get('logisticregression__solver', 'lbfgs') 
        C = hyperparameters.get('logisticregression__C', 1.0)  
        max_iter = hyperparameters.get('logisticregression__max_iter', 1000)
        tol = hyperparameters.get('logisticregression__tol', 1e-4)

        self.mlr = LogisticRegression(multi_class='multinomial', solver=solver, C=C, max_iter=max_iter, tol=tol)
        self.mlr.fit(X, y)

    def predict(self, X):
        return self.mlr.predict(X)