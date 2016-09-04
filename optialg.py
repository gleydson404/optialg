import numpy as np
import matplotlib.pyplot as plt
from alfa_search import calc_alfa


def fx(x):
    '''
    The function to be optimized
    '''
    return (2 * (x[0] - 2)**2) + (2 * (x[1] - 3)**2)


def gradient(x):
    '''
    First devivative of x function computed by hand
    '''
    x1 = 4 * (x[0] - 2)
    x2 = 4 * (x[1] - 3)
    return np.array([x1, x2])


def hessian():
    '''
    Hessiana matrix of fx also computeb by hand
    '''
    return np.array([
                    [4.0, 0.0],
                    [0.0, 4.0]
                    ])


def plot_norm_by_iteration(norm, iterations, title):
    plt.title(title)
    plt.plot(iterations, norm)
    plt.xlabel('Iterations')
    plt.ylabel('Norm')
    plt.show()


def gradient_descendent(x, max_iterations, epslon=1e-3):

    iterations = [0]
    norm = [np.linalg.norm(gradient(x))]
    iteration = 0

    while (iteration <= max_iterations and
           np.linalg.norm(gradient(x)) > epslon):

        iteration += 1
        iterations.append(iteration)

        direction = - gradient(x)
        alfa_best = calc_alfa(x, gradient, direction)
        x = x + (alfa_best * direction)

        norm.append(np.linalg.norm(gradient(x)))

        print("alfa", alfa_best)
        print("iteration", iteration)
        print("norm", norm[iteration])
        print("x", x)

    plot_norm_by_iteration(norm, iterations, "Gradient Descent")


def modified_newton(x, max_iterations, epslon=1e-3):
    iterations = [0]
    norm = [np.linalg.norm(gradient(x))]
    iteration = 0

    def perform_adjustment_law(z, epslon=1e-8):
        '''
        Apply adjusment law wich means:
        if min k th eigen value > 0
            Mk = H(xk)
        else
            Mk = H(xk)+(epslon - min k th eigen value)

        where k means k th valor of
              H means hessiana matrix
              Mk is a way to make sure hessiana is positive

        '''
        hessiana_x = hessian()
        eig_v_min = np.min(np.linalg.eigvalsh(hessiana_x))

        if eig_v_min > 0:
            return hessiana_x
        else:
            # qual o valor de epslon?
            return hessiana_x + (epslon - eig_v_min)

    while (iteration <= max_iterations and
           np.linalg.norm(gradient(x)) > epslon):

        M = perform_adjustment_law(x)

        iteration += 1
        iterations.append(iteration)

        direction = -np.linalg.inv(M).dot(gradient(x))
        alfa_best = calc_alfa(x, gradient, direction)
        x = x + alfa_best * direction

        norm.append(np.linalg.norm(gradient(x)))

        print("alfa", alfa_best)
        print("iteration", iteration)
        print("norm", norm[iteration])
        print("x", x)

    plot_norm_by_iteration(norm, iterations, "Modified Newton")


def levenberg_marquardt(x, max_iterations, epslon=1e-3):
    iterations = [0]
    norm = [np.linalg.norm(gradient(x))]
    iteration = 0

    def generate_r(z):
        '''
        R matrix
        '''
        return np.array([(z[0] - 2), (z[1] - 3)])

    def gradient_r():
        return np.array([
                        [1, 0],
                        [0, 1]
                        ])

    def gradient_f(z):
        r = generate_r(z)
        gr = gradient_r()
        return gr.T.dot(r)

    def calc_hessian_r():
        gr = gradient_r()
        return gr.T.dot(gr)

    while (iteration <= max_iterations and
           np.linalg.norm(gradient(x)) > epslon):

        iteration += 1
        iterations.append(iteration)
        hess = calc_hessian_r()
        direction = -np.linalg.inv(hess).dot(gradient_f(x))
        alfa_best = calc_alfa(x, gradient_f, direction)
        x = x + alfa_best * direction

        norm.append(np.linalg.norm(gradient(x)))

        print("alfa", alfa_best)
        print("iteration", iteration)
        print("norm", norm[iteration])
        print("x", x)

    plot_norm_by_iteration(norm, iterations, "Modified Newton")


def conjugate_gradient(x, max_iterations, epslon=1e-3):
    iterations = [0]
    norm = [np.linalg.norm(gradient(x))]
    iteration = 0
    directions = [- gradient(x)]
    hess = hessian()

    def calc_beta(z, Q, direction):
        return np.divide(gradient(z).T.dot(Q).dot(direction),
                         direction.T.dot(Q).dot(direction))

    def calc_alfa_gc(z, Q, direction):
        return np.divide(direction.T.dot(direction),
                         direction.T.dot(Q).dot(direction))

    while (iteration <= max_iterations and
           np.linalg.norm(gradient(x)) > epslon):

        alfa = calc_alfa_gc(x, hess, directions[iteration])
        x = x + (alfa * directions[iteration])

        beta = calc_beta(x, hess, directions[iteration])

        iteration += 1
        iterations.append(iteration)

        new_direction = -gradient(x) + beta * (directions[iteration - 1])

        directions.append(new_direction)

        norm.append(np.linalg.norm(gradient(x)))

        print("alfa", alfa)
        print("iteration", iteration)
        print("norm", norm[iteration])
        print("x", x)

    plot_norm_by_iteration(norm, iterations, "Gradient Descent")

if __name__ == "__main__":
    '''
    For this function, there is no need to set the number of iterations,
    but if you want this methos to use with other funcionts,
    you have to change the parameter by hand.
    '''
    print("This program shoud to optmize fx(x,y)=2*(x-2)^2+2*(y-3)^2,")
    print("by the following methods:")
    print("1 = Gradient Descent")
    print("2 = Modified Newton")
    print("3 = Levenberg Marquardt")
    print("4 = Conjugate Gradient")
    method = input("Choose your method:")

    X = 4 * np.random.random((2, 1))

    if method == 1:
        gradient_descendent(X, 10)
    elif method == 2:
        modified_newton(X, 10)
    elif method == 3:
        levenberg_marquardt(X, 10)
    else:
        conjugate_gradient(X, 10)
