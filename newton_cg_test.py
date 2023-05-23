import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt


def armijo(f, x, g, d, rho=0.5, c=0.1, alpha=1, **kwargs):
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
    Returns:
        float"""
    f_x = f(x)
    
    gTd = g.dot(d)
    while f(x + alpha*d) > f_x + c*alpha*gTd:
        alpha *= rho
    return alpha


def cg(Q, b, x, tol=1e-9, max_it = 30000, callback=None):    
    """Conjugate gradient algorithm
    
    Solves the problem 
    min xQx/2. + xb
    
    Parameters:
        Q: ndarray, positive definite, symmetric matrix Q
        b: ndarray, negative right side
        x: ndarray, start value
        tol: [float], gradient norm tolerance
        maxit: [int], maximum number of iterations
        callback: [callable], Callback function
    Returns:
        ndarray, solution
    """
    n = b.size    
    beta = 0
    p = np.zeros(n)
    r = Q@x + b
    res_new  = norm(r)
    k=0   
    while res_new >= tol and k < max_it:
        k+=1
        p = -r + beta*p
        alpha = res_new**2 / p.dot(Q@p)
        if not callback is None:
            callback(x)
        x = x + alpha*p
        r = r + alpha*(Q@p)
        res_old = res_new
        res_new = norm(r)
        beta = res_new**2/res_old**2
    if not callback is None:
        callback(x)
    if k==max_it:
        print("Algorithm reached max iterations!")
    return  x


b=100
def f(z):
    """Rosenbrock function
    Parameters:
        z: nd_array, 2-D input value
    Returns:
        float"""
    x,y = z
    return (1-x)**2 + b*(y- x**2)**2


def df(z):
    """First derivative of Rosenbrock function
    Parameters:
        z: nd_array, 2-D input value
    Returns:
        nd_array, 2D vector of partial derivatives"""
    x,y = z
    dx = -2*(1-x) - 4*b*(y-x**2)*x
    dy = 2*b*(y- x**2)
    return np.array([dx, dy])


def Hessf(z):
    """Second derivative of Rosenbrock function
    Parameters:
        z: nd_array, 2-D input value
    Returns:
        nd_array of shape (2,2), matrix containing second derivatives"""
    x,y = z
    dxx = 2 - 4*b*(y-3*x**2)
    dyx = -4*b*x
    dyy = 2*b
    return np.array([[dxx, dyx], [dyx, dyy]])


class CallBack:
    """Call back
    
    Collects information aboute the iterates xk in a list 
    self.xk.
    """
    def __init__(self, x0=None):
        # The Class constructor is executed when 
        # we create an new isntance by obj = CallBack().
        # It takes no arguments.
        if x0 is not None:
            self.xk=[x0.copy()]
        else:
            self.xk = []

    def __call__(self, xk):
        # The means an object obj = CallBack
        # can be executed like a function by obj().
        self.xk.append(xk.copy())
        return False
    def getxk(self):
        return np.array(self.xk)
    
    def plot(self, f, xmin=-1.5, xmax=1.5):
        
        Xk = np.array(self.xk)
        l = np.arange(xmin,xmax,.01)
        X,Y = np.meshgrid(l,l)
        XY = np.vstack([X.ravel(),Y.ravel()]).T
        Z = np.array([f(xy) for xy in XY])
        Z=Z.reshape(X.shape)
        plt.contourf(X,Y,Z,levels=15)
        plt.contour(X,Y,Z,levels=10)
        zk = np.array([f(xk) for xk in self.xk])
        plt.scatter(Xk[:,0], Xk[:,1], c="white", s=15)
        plt.plot(Xk[:,0], Xk[:,1], color="white", alpha=.6)
        #plt.scatter(0,0,c="r")
        plt.show()

    def error_plot(self, true_solution):
        errors = np.array([np.linalg.norm(xk - true_solution) for xk in self.xk])
        plt.plot(errors)
        plt.yscale("log")
        plt.title("Norm of Errors")
        plt.xlabel("iteration")
        plt.ylabel("error (log-scale)")
        

def print_dict(dictionary):
	[print(key, ": \t\t", dictionary[key]) for key in dictionary]


def test(algorithm):
	x0 = np.array([0.1, 0.1])
	callback = CallBack(x0)
	sol = algorithm(f, df, Hessf, x0, callback=callback)
	callback.plot(f)
	callback.error_plot(np.array([1., 1.]))
	#print_dict(sol)




