import numpy as np


def calc_hl(x, alfa, direction, gradient):
    x_new = x + (alfa * direction)
    gradient_xnew = gradient(x_new)
    return gradient_xnew.T.dot(direction).item(0)


def is_hl_zero(hl):
        if np.abs(hl) < 1e-8:
            return True
        return False


def find_hl(x, direction, alfa, gradient):
    hl = calc_hl(x, alfa, direction, gradient)
    while hl < 0:
        alfa = 2 * alfa
        hl = calc_hl(x, alfa, direction, gradient)
    return hl, alfa


def calc_num_iterations(alfa, epslon):
        return np.ceil(np.log(alfa/epslon))


def calc_alfa(x, gradient, direction):
    x_old = x
    # direction = - gradient(x_old)
    alfa_low = 0
    alfa_hight = np.random.random_sample()
    epslon = 1e-8
    lbda = 1e-8

    hl = calc_hl(x_old, alfa_hight, direction, gradient)

    if is_hl_zero(hl):
        return alfa_hight

    hl, alfa_hight = find_hl(x_old, direction, alfa_hight, gradient)

    if is_hl_zero(hl):
        return alfa_hight

    num_iterations = calc_num_iterations(alfa_hight, epslon)
    iteration = 0
    mean_alfa = (alfa_hight + alfa_low)/2

    while (iteration < num_iterations and
           np.abs(hl) > lbda):
        iteration += 1
        mean_alfa = (alfa_hight + alfa_low)/2
        hl = calc_hl(x_old, mean_alfa, direction, gradient)
        if hl > 0:
            alfa_hight = mean_alfa
        elif hl < 0:
            alfa_low = mean_alfa
        else:
            break
    return mean_alfa
