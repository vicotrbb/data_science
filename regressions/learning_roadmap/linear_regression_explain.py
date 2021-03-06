# Regressao Linear
# by: Victor Bona
# Fonte: https://towardsdatascience.com/linear-regression-using-python-b136c91bf0a2

# A regressao linear normalmente e o primeiro algoritmo de machine learning que os cientistas de dados aprendem, 
# apesar de simples e muito importante masterizar esse metodo, visto que muitos problemas igualmente simples mas importantes 
# podem ser resolvidos atraves dele.

# Onde a Regressao linear pode ser utilizada?

# A regressao linear e uma tecninca muito simples, mas poderosa, pode ser utilizada para entender e medir os fatores que influenciam
# a rentabilidade. Pode ser usado para prever precos de produtos baseado em seu historico de precos, rastrear comportamento de 
# clientes, entre outras coisas.

# Oque e exatamente a Regressao linear?

# O objetivo da regressao linear e encontrar um relacionamento entre uma ou mais variaveis independentes e caracteristicas 
# constantes de um conjunto de dados, basicamente, encontrar dependencias em uma variavel aparentemente independente. 
# Pode ser de 2 tipos: 
# Univariate Linear Regression: Quando so existe uma variavel independente.
# Multiple Linear Regression: Quando existem mais variaveis indepentendes.

# A hipotese da regressao linear

# O modelo da regressao linear pode ser representado pelo seguinte modelo matematico:
#   Y = θ₀ + θ₁x₁ + ... + θₓxₓ

# Y = O valor previsto
# θ₀ = O termo de bias(desvio sistemático do valor real)
# θ₁ ... θₓ = Os paraemtros do modelo
# x₁ ... xₓ = Variveis independentes
# Lembrando que o termo de bias representa a variacao sistematica do valor estimado para o valor real.

# A Hipotese pode ser representada tambem por: Y = θⁿx
# θⁿ = Inclui o vetor de parametros do modelo e do termo de bias
# x = O vetor de variaveis independes, sendo que x₁ = 1

# Criacao de um dataset de testes, gerado randomicamente
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)

# ------------------------------------------------------------

# Processo de treino do modelo de regressao linear

# Treinar o modelo significa encontrar o conjunto de parametros que melhor se encaixa nos dados.
# Como determinamos a melhor linha para encaixarmos nosso dados?

# Para determinarmos a melhor linha de previsao, precisamos procurar alcancar a menor taxa residual(erro), residuos sao a diferenca 
# entre o valor predito e o valor observado, representado pela funcao: valor residual = valor observado - valor predito
# A linha formada pelos menores valores residuais e chamada de linha de regressao.
# Para medirmos a taxa de erro de nosso modelo, definimos a seguinte funcao de custo: A soma dos quadrados de todos os residuos:

# J(θ) = 1/2(Σ(h(xᶦ) - yᶦ)²)
# Sendo que i=1

# E a funcao de hipotese e representada por: 

# h(x) = θ₀ + θ₁x₁ + ... + θₓxₓ
# Sendo que, m = numero de exemplos de treino em nosso modelo

# O motivo de usarmos o quadrado dos residuos e por que queremos penalizar os residuos mais distantes da linha de regressao e 
# beneficiar os pontos que se encontram mais perto dela.

# Nosso objetivo e encontrar os parametros adequados para que nossa funcao de custo seja minima, para isso, utilizamos o 
# Implementando o modelo de regressao linear utilizando o método do gradiente

# Oque e o metodo do gradiente

# o metodo do gradiente um algoritmo de otimizacao generico, utilizado em varios algoritimos de machine learning. 
# Ele realiza o trabalho de ajustar os parametros do modelo dinamicamente visando abaixar a funcao de custo / taxa de erro

# Como funciona o metodo do gradiente

# 1 - Primeiro inicializamos o modelo com parametros aleatorios, chamamos isso de inicializacao randomica.
# 2 - Medimos as mudancas na funcao de custo realizada pela mudanca de parametros, portanto calculamos as derivadas parciais
# da funcao de custo para os parametros θ₀, θ₁, ..., θₓ, descrito pela funcao:
# δJ(θ)/δθⱼ = 1/(Σ(h(xᶦ) - yᶦ)xᶦⱼ)
# Sendo que i=1
# h(x) = θ₀ + θ₁x₁ + ... + θᵤxᵤ
# E possivel calcular a derivativa parcial de todos os parametros ao mesmo tempo, vide https://towardsdatascience.com/linear-regression-using-python-b136c91bf0a2

# 3 - Depois de calcular a derivativa, precisamos atualizar todos os paraemtros
# θᵤ = a/(Σ(h(xᶦ) - yᶦ)xᶦᵤ)
# Sendo que a = Parametro de aprendizado.
# E possivel atualizar todos os paraemtros ao mesmo tempo, vide https://towardsdatascience.com/linear-regression-using-python-b136c91bf0a2

# Para treinamos o modelo, os processos 2 e 3 sao repetidos varias vezes ate que a funcao de custo converja para o menor
# valor possivel.
# A taxa de aprendizado precisa ser definida com cuidado, visto que se for menor demais, o modelo vai demorar demais para ser
# treinado. Caso seja muito grande, e posssivel que o modelo falhe ao convergir para o minimo da funcao de custo e passar dela

# Implementando o algoritmo 


class LinearRession:

    """
    Parametros:
        eta: float, taxa de aprendizado
        n_iterations: int, numero de iteracoes do conjunto de treinamento

    Atributos:
        w: pesos
        cost: erro do modelo apos cada iteracao
    """

    def __init__(self, eta=0.05, n_iterations=1000):
        self.eta = eta # eta representa a taxa de aprendizado do modelo
        self.n_iterations = n_iterations # Representa o numero de iteracoes que o melo vai executar durante o aprendizado
        self.cost_ = [] 
        self.w_ = []

    def fit(self, x, y):
        """
        Parametros:
        x: array, [amostras, recurso]
            Amostras de treino
        y: array, [amostras, valores alvo]
            Valores alvo
        Retorna:
            O proprio objeto
        """
        self.w_ = np.zeros((x.shape[1], 1))  # np.zeros retorna um array de 0, 
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
        m = x.shape[0]

        for _ in range(self.n_iterations):
            y_pred = np.dot(x, self.w_) #  np.dot retorna o produto de 2 arrays, 
            #https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
            residuals = y_pred - y
            gradient_vector = np.dot(x.T, residuals)
            self.w_ -= (self.eta / m) * gradient_vector
            cost = np.sum((residuals ** 2)) / (2 * m) # Calcula a funcao de custo
            self.cost_.append(cost)
        return self

    def predict(self, x):
        """
        Parametros:
        x: array, [amostras, recurso]
            Amostras de treino
        Retorna:
            Valor predizido
        """
        return np.dot(x, self.w_)

    def evaluate(y, y_predicted):
        """
        Parametros:
        y: array, [amostras, recurso]
            Amostras de treino
        y_predicted: array, [amostras, valores preditos]
            Valores preditos pelo modelo
        Retorna:
            Acuracia do modelo
        """
        mse = np.sum((y_predicted - y)**2)
        mrse = np.sqrt(mse/len(y))

        ssr = np.sum((y_predicted - y)**2)
        sst = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ssr/sst)
        return mrse, r2

# Processo de avalicao de acuracia do algortimo

# Raiz da media do quadrado da soma dos residuos
# Este metodo e definido pela seguinte funcao:
# √(1/(Σ(h(xᶦ) - yᶦ)²))

# Coeficiente de determinação
# Este metodo e composto pelo coeficiente da terminacao, representado pela funcao
# R² = 1 - ssr/sst
# Sendo que, ssr = Σ(h(xᶦ) - yᶦ)²)
#           sst = Σ((yᶦ - y_ᶦ)²)


# ---------------------------------------------------IMPLEMENTACAO SKLEARN--------------------------------------------------------

# O scikit-learn(sklearn) e uma biblioteca de aprendizado de maquina de codigo aberto para o python.
# Exemplo de implementacao detalhada de uma regressao linear usando o sklearn

# Primeiro separamos nosso conjunto de dados em 2, um de treino e outro de teste.
# Fazemos isso para avaliar a perfomance de nosso modelo, dividimos o modelo em 80% para treino e 20% para testes
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

# Criamos o modelo e treinamos ele com nosso conjunto de treino
model = LinearRegression()
model.fit(x_train, y_train) # X conteem os parametros 'LSTAT' e 'RM'. Y conteem o preco das casas 

# Realizamos a previsao e medimos a acuracia de nosso conjunto de treino
# Conjunto de treino
train_predict = model.predict(x_train) # Para testarmos a eficacia de nosso modelo com dados conhecidos, usamos o conjunto de
# treino para avaliarmos.
rmse = (np.sqrt(mean_squared_error(y_train, train_predict))) # Tiramos as metricas de perfomance dele, usando os metodos
# mean_squared_error & r2_score
r2 = r2_score(y_train, train_predict)
plt.scatter(y_train, train_predict, s=10, c='red', label='Teste') # Plotamos o resultado para futura comparacao

print('Train set perfomance')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')

# Conjunto de teste
# Realizamos agora uma avaliacao com nosso conjunto de testes, esses dados ainda nao foram apresentados ao nosso modelo, por 
# isso a importancia desse teste, para termos nocao de como o modelo vai se comportar com dados desconhecidos
test_predict = model.predict(x_test)
rmse = (np.sqrt(mean_squared_error(y_test, test_predict)))
r2 = r2_score(y_test, test_predict)
plt.scatter(y_test, test_predict, s=10, c='blue', label='Treino') # Plotamos o resultado para comparacao

print('Train set perfomance')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')

# Plotamos ambos os resultados em um grafico
plt.legend(loc='lower right')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

