# Implementacao de um modelo de regressao linear em um dataset de casas em boston. O objetivo e prever o preco das casas
# by: Victor Bona
# Fonte: https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155

# Implementacao utilizando o sklearn

import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns 

# O dataset esta disponivel entre os datasets do skelarn.
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

# Criamos uma nova coluna 'Target' baseada nos valores alvo das casas
boston['Target'] = boston_dataset.target

# O dataset nao possui valores vazios, entao nao existe a necessidade de tratar isso.
# Preparando os dados para o modelo de treino

# Apos analise dos dados do dataset, percebemos uma forte correlacao dos valores das casas com os atributos 'LSTAT' e 'RM', por isso
# utilizamos esses atributos como nossos parametros
x = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns=['LSTAT', 'RM'])
y = boston['Target']

# Precisamos separar nosso dataset em conjuntos de treino e de teste, para isso, usamos a funcao 'train_test_split' do sklearn,
# Usamos 80% do conjunto para treino e 20% para testes
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

# Criamos o modelo
model = LinearRegression()
model.fit(x_train, y_train)

# Realizamos a previsao e medimos a acuracia de nosso conjunto de treino
# Conjunto de treino
train_predict = model.predict(x_train)
rmse = (np.sqrt(mean_squared_error(y_train, train_predict)))
r2 = r2_score(y_train, train_predict)
plt.scatter(y_train, train_predict, s=10, c='red', label='Teste')

print('Train set perfomance')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')

# Conjunto de teste
test_predict = model.predict(x_test)
rmse = (np.sqrt(mean_squared_error(y_test, test_predict)))
r2 = r2_score(y_test, test_predict)
plt.scatter(y_test, test_predict, s=10, c='blue', label='Treino')

print('Train set perfomance')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')

# Plotamos ambos os resultados em um grafico
plt.legend(loc='lower right')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
