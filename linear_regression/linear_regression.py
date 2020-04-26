import numpy as np

# Explicacao em linear_regression/linear_regression_explain.py

class LinearRegression:

    def __init__(self, eta=0.05, n_iterations=1000):
        self.eta = eta 
        self.n_iterations = n_iterations
        self.cost_ = [] 
        self.w_ = []

    def fit(self, x, y):
        self.w_ = np.zeros((x.shape[1], 1))  
        m = x.shape[0]

        for _ in range(self.n_iterations):
            y_pred = np.dot(x, self.w_)
            residuals = y_pred - y
            gradient_vector = np.dot(x.T, residuals)
            self.w_ -= (self.eta / m) * gradient_vector
            cost = np.sum((residuals ** 2)) / (2 * m)
            self.cost_.append(cost)
        return self

    def predict(self, x):
        return np.dot(x, self.w_)

    def evaluate(self, y, y_predicted):
        mse = np.sum((y_predicted - y)**2)
        mrse = np.sqrt(mse/len(y))

        ssr = np.sum((y_predicted - y)**2)
        sst = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ssr/sst)
        return mrse, r2
