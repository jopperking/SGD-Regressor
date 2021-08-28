from sklearn.linear_model import SGDRegressor
from sklearn.datasets import make_regression


#dataset loading
x,y = make_regression(n_samples=186,n_features=5,n_targets=1)
print(x)


#Model Learning
sgd_reg = SGDRegressor(max_iter=100)
print(sgd_reg)

sgd_reg.fit(x,y)
print(sgd_reg.coef_)

#model eval
result = sgd_reg.predict([x[67]])
print(result)
print(y[67])
