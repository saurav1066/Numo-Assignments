import numpy as np
import matplotlib.pyplot as plt

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
        
def test_nlcg(nlcg):
    print("\n# Test nlcg on Rosenbrock function.")
    def frosenbrock(z):
        """Rosenbrock function
        Parameters:
            z: nd_array, 2-D input value
        Returns:
            float"""
        x,y = z
        return (1-x)**2 + 100*(y- x**2)**2

    #callback=CallBack()
    x0=np.zeros(2)

    #sol_nlcg = nlcg(frosenbrock,x0,callback=callback)
    sol_nlcg = nlcg(frosenbrock,x0)
    
    #print("Iterations: ", callback.getxk().shape[0])
    #print("Iterates:\n", callback.getxk())
    print("Test (should be 0):\n", frosenbrock(sol_nlcg))
    #callback.plot(frosenbrock)

def test_cg(cg):
    print("\n# Test cg on xQx+b.")
    Q = np.array([[1, 2], [2, 10]])/2
    print("Matrix Q\n\n", Q)
    b = np.array([1.,1.])
    print("\nRight hand side -b\n", -b)
    x0 = np.array([0.,0.])

    def fQ(x):
        return x@Q@x/2 + b@x
    
    #callback=CallBack()
    #sol_cg = cg(Q,b,x0,callback=callback)
    sol_cg = cg(Q,b,x0)
    
    #print("Iterations: ", callback.getxk().shape[0])
    #print("Iterates:\n", callback.getxk())
    print("\nTest (should be equal to -b):\nQx =", Q@sol_cg)
    #callback.plot(lambda x: x@Q@x/2 + b@x, xmin=-4, xmax=2.)

def test(fun):
    dict = {"cg": test_cg, "nlcg": test_nlcg}
    return dict[fun.__name__](fun)

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