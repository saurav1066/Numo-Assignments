import numpy as np
import matplotlib.pyplot as plt


def sigmoidal(t):
	return 1/(1 + np.exp(-t))


def dsigmoidal(t):
	s = sigmoidal(t)
	return s * (1 - s)


def generate_data(gamma=0.0, seed=99):
	np.random.seed(seed)
	n = 10
	t = np.random.rand(n) - 0.5
	noise = np.random.randn(n)*gamma
	signal = sigmoidal(t*6. - 1.)
	alpha = signal + noise
	return t, alpha


def generate_probabilities(gamma=0.0, seed=99):
	np.random.seed(seed)
	n = 10
	t = np.random.rand(n) - 0.5
	noise = np.random.rand(n)*gamma
	signal = sigmoidal(t*6. + 2.)*(1 - noise)
	alpha = signal
	return t, alpha


def plot(x, gamma):
	t, alpha = generate_data(gamma)
	t_axis = np.linspace(-3, 3, 50)
	plt.scatter(t, alpha)
	plt.plot(t_axis, sigmoidal(t_axis*x[0] + x[1]))
	plt.show()


def convergence_plot(delta_x_list):
	plt.plot(delta_x_list)
	plt.yscale("log")
	plt.show()


def armijo(f, x, g, d, rho=0.5, c=0.01, alpha=1, **kwargs):
	"""Armijo line search 

	Parameters:
		f: callable, Function to be minimized
		x: ndarray, current iterate x
		g: ndarray, current gradient of f() at x
		d: ndarray, current descent direction, e.g. -g
		rho: [float], scaling factor of stepsize
		c: [float], factor in minimum decrease condition
		alpha: [float], start step size
		**kwargs: [keyword args] 
	Returns: float"""
	f_x = f(x)

	gTd = g.dot(d)
	while f(x + alpha*d) > f_x + c*alpha*gTd:
		alpha *= rho
	return alpha





