import quadprog
import numpy as np

def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -numpy.vstack([A, G]).T
        qp_b = -numpy.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

def skew_3d(omega):
    """
    Converts a rotation vector in 3D to its corresponding skew-symmetric matrix.

    Args:
    omega - (3,) ndarray: the rotation vector

    Returns:
    omega_hat - (3,3) ndarray: the corresponding skew symmetric matrix
    """
    if not omega.shape == (3,):
        raise TypeError('omega must be a 3-vector')

    omega_hat = np.zeros((3,3))
    omega_hat[0,2] = omega[1]
    omega_hat[2,0] = -omega[1]
    omega_hat[1,0] = omega[2]
    omega_hat[0,1] = -omega[2]
    omega_hat[2,1] = omega[0]
    omega_hat[1,2] = -omega[0]

    return omega_hat


def rot_x(x):
    cos_x = np.cos(x)
    sin_x = np.sin(x)
    return np.array([[1, 0, 0],[0 cos_x, -sin_x],[0, sin_x, cos_x]])

def rot_y(y):
    cos_y = np.cos(y)
    sin_y = np.sin(y)
    return np.array([[cos_y, 0, sin_y],[0, 1, 0],[-sin_y, 0, cos_y]])

def rot_z(z):
    cos_z = np.cos(z)
    sin_z = np.sin(z)
    return np.array([[cos_z, -sin_z, 0],[sin_z, cos_z, 0],[0, 0, 1]])

def rot_rpy(r, p, y):
    return rot_z(y).dot(rot_y(p).dot(rot_x(r)))
