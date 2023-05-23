import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test(fun):
    dict = {"drv": test_drv, "ddrv": test_ddrv, "pdrv": test_pdrv, "grad": test_grad, "hess": test_hess}
    return dict[fun.__name__](fun)


def test_drv(drv):
    # Function of which we want to compute the derivative
    def f(x):
        return np.sin(x**2)

    # analytical solution
    def f_analytical_drv(x):
        return 2*x * np.cos(x**2)

    h = 1e-6

    # Evaluate function, its derivative and the numerical derivative
    x_values = np.array(np.linspace(0, 5, 1000), dtype=np.float64)
    function_values = [f(x) for x in x_values]
    analytical_drv = [f_analytical_drv(x) for x in x_values]
    numerical_drv = [drv(f, x, h) for x in x_values]

    # Plot graph of function
    plt.figure("Function f")  # new canvas
    plt.plot(x_values, function_values, label="f")  # plt graph
    plt.legend()  # plot legend

    # Plot first derivative
    plt.figure("First derivative")
    plt.plot(x_values, analytical_drv, label="f' analytical")
    plt.plot(x_values, numerical_drv, label="f' numerical")
    plt.legend()

    # print Errors
    plt.figure("Error f'")
    errors = [np.abs(drv(f, x, h) - f_analytical_drv(x)) for x in x_values]
    plt.plot(x_values, errors, label="Error")
    plt.legend()

    # Show plots
    # plt.show()


def test_ddrv(ddrv):
    # Function of which we want to compute the derivative
    def f(x):
        return np.sin(x**2)

    # analytical solution
    def f_analytical_ddrv(x):
        return 2*np.cos(x**2) - 4*x**2*np.sin(x**2)

    h = 1e-4

    # Evaluate function, its derivative and the numerical derivative
    x_values = np.array(np.linspace(0, 5, 1000), dtype=np.float64)
    function_values = [f(x) for x in x_values]
    analytical_ddrv = [f_analytical_ddrv(x) for x in x_values]
    numerical_ddrv = [ddrv(f, x, h) for x in x_values]

    # Plot graph of function
    plt.figure("Function f")  # new canvas
    plt.plot(x_values, function_values, label="f")  # plt graph
    plt.legend()  # plot legend

    # Plot first derivative
    plt.figure("First derivative")
    plt.plot(x_values, analytical_ddrv, label="f' analytical")
    plt.plot(x_values, numerical_ddrv, label="f' numerical")
    plt.legend()

    # print Errors
    plt.figure("Error f'")
    errors = [np.abs(ddrv(f, x, h) - f_analytical_ddrv(x)) for x in x_values]
    plt.plot(x_values, errors, label="Error")
    plt.legend()

    # Show plots
    plt.show()


# Hilfsfunktion zum plotten einer Funktion f:R^2 -> R
def my_plot(f, N=100, x_min=0., x_max=1., y_min=0., y_max=1.):
    """Plot the function values of f on a 2-D grid."""
    # Create mesh based on points x and y.
    x = np.linspace(x_min, x_max, N)
    y = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid(x, y)

    # Compute function values
    F = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            F[i, j] = f(np.array([x[i], y[j]]))

    # return plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, F)


def test_grad(grad):
    # Function
    def g(x):
        return (x[0]-0.5)**2 + np.sin(2*np.pi*x[1])

    # Gradient analytical
    def g_analytical_grad(x):
        return np.array([2*x[0] - 1., 2*np.pi*np.cos(2*np.pi*x[1])])

    # Distance between analytical and numerical gradient
    def error(x):
        return np.linalg.norm(g_analytical_grad(x) - grad(g, x, 1e-6))

    # Plot function
    my_plot(g)
    # Plot error
    my_plot(error)
    plt.show()


def test_pdrv(pdrv):
    # Function
    def g(x):
        return (x[0]-0.5)**2 + np.sin(2*np.pi*x[1])

    # Analytical derivative
    def g_analytical_drv1(x):
        return 2.*x[0] - 1.

    def error(x):
        return np.abs(g_analytical_drv1(x) - pdrv(g, x, 0, 1e-6))

    # Plot function
    my_plot(g)
    # Plot error
    my_plot(error)
    plt.show()

def test_hess(hess):
    # Function
    def g(z):
        x,y = z
        return (x-0.5)**2 + np.sin(2*np.pi*y)

    # Analytical d^2/dx^2 + d^2/dy^2
    def g_analytical_laplace(z):
        x,y = z
        ddx = 2
        ddy = -np.sin(2*np.pi*y)*np.pi**2*4
        return ddx + ddy

    def error(x):
        num_laplace = np.sum(np.diag(hess(g, x, 1e-6)))
        return np.abs(g_analytical_laplace(x) - num_laplace )

    h = 1e-6

    my_plot(g)

    my_plot(error)
    # Show plots
    plt.show()
