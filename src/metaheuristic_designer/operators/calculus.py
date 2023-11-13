import numpy as np

def partial_der(f, pos, dir, delta=1e-10):
    """
    Symmetric derivative
    """

    step = delta*dir 
    return (f(pos + step) - f(pos - step))/(2*delta)

def directional_der(f, pos, dir):
    dir_norm = (dir**2).sum()
    return partial_der(f, pos, dir/dir_norm)

def second_partial_der(f, pos, dir1, dir2, delta=1e-3):
    x_diff = f(pos - delta*(dir1 + dir2)) - f(pos - delta*(dir1 - dir2))
    y_diff = f(pos + delta*(dir1 - dir2)) - f(pos + delta*(dir1 + dir2))
    return (x_diff - y_diff)/(delta*delta)

# Scalar valued functions
def gradient(f, pos, delta=1e-10):
    pos = np.asarray(pos)
    grad = np.empty(pos.size)
    basis =  np.eye(pos.size)

    for i, base in enumerate(basis):
        grad[i] = partial_der(f, pos, base, delta=1e-10)
    return grad

def hessian(f, pos):
    pos = np.asarray(pos)
    hess = np.empty([pos.size, pos.size])
    basis =  np.eye(pos.size)

    for i, basex in enumerate(basis):
        for j, basey in enumerate(basis):
            hess[i,j] = second_partial_der(f, pos, basex, basey)
    return hess

# Vector values functions
def jacobian(f, pos):
    pos = np.asarray(pos)
    jac = np.empty([f(pos).size, pos.size])
    basis =  np.eye(pos.size)

    for base in basis:
        jac[:,i] = partial_der(f, pos, base)
    return jac

def divergence(f, pos):
    return np.trace(jacobian(f, pos))