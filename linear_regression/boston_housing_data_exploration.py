# Execucao da analise exploratoria de dados para o dataset 'Boston_housing' do sklearn
# by: Victor Bona
# Fonte: https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155
# Este arquivo serve apenas para estudos no roadmap sobre regressao linear, se quiser fazer uso dele, recomendo executa-lo em 
# Jupyter notebook, visto que nele, os graficos se sobrepoem.
# Ao final do roadmap, sera disponbilizado um jupyter notebook com todas as informacoes compiladas.

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns 

# O dataset esta disponivel entre os datasets do skelarn.
from sklearn.datasets import load_boston

boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

# Criamos uma nova coluna 'Target' baseada nos valores alvo das casas
boston['Target'] = boston_dataset.target

# Verificamos se o dataset possui valores vazios
print('Valores vazios: ')
print(boston.isnull().sum())

# Primeiro entemos como esta distribuido nosso valor alvo das casas, para isso, utilizamos o matplotlib e o seaborn para plotarmos 
# um grafico de barras e linha.
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.distplot(boston['Target'], bins=30)
plt.show()

# Identificado que a distribuicao esta normal, com apenas algumas anormalidas, precisamos entao, encontrar a correlacao entre nossos
# atributos, para isso, utilizamos uma matriz de correlacionamento.
plt.clf()
matrix_correlation = boston.corr().round(2)
sns.heatmap(data=matrix_correlation, annot=True)
plt.show()

# Na Matriz de correlacao ela pode ser negativa ou positiva e vai de -1 a 1, quanto mais perto de 1, mais forte a correlacao 
# positiva dos atributos e quanto mais proxima de -1, maior a correlacao negativa dos atributos

# Nesse caso, identificamos que os atributos 'LSTAT' e 'RM' possuem forte correlacao com o nosso atributo alvo, sendo o 'LSTAT' 
# uma correlacao positiva e o 'RM' negativa. Sendo assim, identificamos que o mais acertivo seria utilizar esses atributos para a 
# regressao

# Adicionalmente, e possivel verificar de que forma esses atributos interferem no atributo alvo, para isso, podemos utilizar um
# grafico de dispersao para melhor visualizacao.

plt.clf()
plt.figure(figsize=(20,5))
features = ['LSTAT', 'RM']
target = boston['Target']

for i, col in enumerate(features):
	plt.subplot(1, len(features), i+1)
	x = boston[col]
	y = target
	plt.scatter(x, y)
	plt.title(col)
	plt.xlabel(col)
	plt.ylabel('Target')
plt.show()


