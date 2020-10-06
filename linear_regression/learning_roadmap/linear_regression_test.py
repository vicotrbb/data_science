import linear_regression
import matplotlib.pyplot as plt
import numpy as np

# Gerando um dataset para testes
np.random.seed(0)
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)

# Teste da minha implementacao

model = linear_regression.SimpleLinearRegression()

model.fit(x, y)
predicted = model.predict(x)

mrse,r2 = model.evaluate(y, predicted)
print(f'mrse: {mrse}')
print(f'r2: {r2}')

plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x, predicted, color='r')
plt.show()
