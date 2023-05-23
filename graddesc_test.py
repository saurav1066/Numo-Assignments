import numpy as np
import matplotlib.pyplot as plt

def pdrv(f, x, i, h=1e-6):
    """
    Compute the approximate partial derivative of a function with cetral differences.
    
    Parameters:
        f: callable, Function R^n -> R of which we compute the partial derivative
        x: nd_array, Point at which the derivative is evaluated
        i: 0,...,n-1, Index of differentiation variable x[i]
        h: double,   Stepsize
    Returns:
        double, Approximate partial derivative
    """
    # Build step
    step = np.zeros(np.size(x))
    step[i] = h
    # step = (0,0,0,...,h,....,0,0,0)
    # directional derivative
    return (f(x+step) - f(x-step))/(2.*h)
# Some of you encountered a problem which is related to the difference 
# between copies and views, or references and values. You find some information here.
# https://www.jessicayung.com/numpy-views-vs-copies-avoiding-costly-mistakes/


def grad(f, x, h=1e-6):
    """
    Compute the approximate gradient (w.r.t. standard Euclidean scalar product)
    of a function.
    
    Parameters:
        f: callable, Function R^n -> R of which we compute the derivative
        x: nd_array, Point at which the derivative is evaluated
        h: double,   Stepsize
    Returns:
        double, Approximate gradient 
    """
    # read number of dimensions n
    n = np.size(x)

    # initialize gradient vector
    # type ndarray
    # (1.2, 3, 4.5, 4.3, 0.0, ...., 4.3) of length n
    grad = np.zeros(n)

    # compute partial derivatives in each direction i=0,...,n-1
    for i in range(n):
        grad[i] = pdrv(f, x, i, h)

    return grad

class CallBack:
    """Call back
    
    Collects information aboute the iterates xk in a list 
    self.xk.
    """
    def __init__(self):
        # The Class constructor is executed when 
        # we create an new isntance by obj = CallBack().
        # It takes no arguments.
        self.xk=[]
        
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

Q = np.array([[5, 1],[1, 50]])
def f(x):
    return x@Q@x

def test(gradientdescent):
    x0 = np.ones(2)
    method = ["constant", "armijo", "wolfe"]
    
    for m in method:
        try:
            solution = gradientdescent(f, x0, method=m )
            print("Method: ", m, "\tx = ", np.round(solution, 4))
        except:
            print("Method", m, "does not work.")
            raise