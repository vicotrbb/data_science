# Usado para gerar datasets de teste de forma randomica
import numpy as np
import matplotlib.pyplot as plt


class Misc:

	def generate_random_ds():
		np.random.seed(0)
		x = np.random.rand(100, 1)
		y = 2 + 3 * x + np.random.rand(100, 1)
		return x, y
