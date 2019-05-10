import quadprog
import numpy as np

def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
   qp_G = .5 * (P + P.T)   # make sure P is symmetric
   qp_a = -q
   if A is not None:
       qp_C = -np.vstack([A, G]).T
       qp_b = -np.hstack([b, h])
       meq = A.shape[0]
   else:  # no equality constraint
       qp_C = -G.T
       qp_b = -h
       meq = 0
   return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

def rotation_3d(omega, theta):
    """
    Computes a 3D rotation matrix given a rotation axis and angle of rotation.
    Args:
    omega - (3,) ndarray: the axis of rotation
    theta: the angle of rotation
    Returns:
    rot - (3,3) ndarray: the resulting rotation matrix
    """
    if not omega.shape == (3,):
        raise TypeError('omega must be a 3-vector')

    omega_norm = np.linalg.norm(omega)
    rot = np.eye(3) + skew_3d(omega) / omega_norm * np.sin(omega_norm * theta) + \
        np.dot(skew_3d(omega),skew_3d(omega)) / (omega_norm*omega_norm) * (1 - np.cos(omega_norm * theta))

    # Alternative method
    #rot = sp.linalg.expm(skew_3d(omega)*theta)
    return rot

def homog_3d(xi, theta):
    """
    Computes a 4x4 homogeneous transformation matrix given a 3D twist and a 
    joint displacement.
    Args:
    xi - (6,) ndarray: the 3D twist
    theta: the joint displacement
    Returns:
    g - (4,4) ndarary: the resulting homogeneous transformation matrix
    """
    if not xi.shape == (6,):
        raise TypeError('xi must be a 6-vector')

    g = np.zeros((4,4))
    v = xi[0:3][...,np.newaxis]
    w = xi[3:6]
    R = rotation_3d(w, theta)
    w_skew = skew_3d(w)
    w = xi[3:6][...,np.newaxis]
    w_skewnorm = np.linalg.norm(w)
    p = 1 / np.power(w_skewnorm,2) * (np.dot(np.eye(3) - R, np.dot(w_skew,v)) + \
         np.dot(np.dot(w, np.transpose(w)),v * theta))
    g[0:3,0:3] = R
    g[0:3,3:4] = p
    g[3,3] = 1

    # Alternative method
    #g = sp.linalg.expm(hat_3d(xi)*theta)

    return g

def prod_exp(xi, theta):
    """
    Computes the product of exponentials for a kinematic chain, given
    the twists and displacements for each joint.

    Args:
    xi - (6,N) ndarray: the twists for each joint
    theta - (N,) ndarray: the displacement of each joint
    Returns:
    g - (4,4) ndarray: the resulting homogeneous transformation matrix
    """
    if not xi.shape[0] == 6:
        raise TypeError('xi must be a 6xN')

    g = np.eye(4)
    for i in range(0, xi.shape[1]):
        g_i = homog_3d(xi[:,i], theta[i])
        g = np.dot(g, g_i)

    return g

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

def quat_to_rot(quat):
    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]
    return np.array([[1.0 - 2.0*y**2 - 2.0*z**2 ,       2.0*x*y - 2.0*z*w       , 2.0*x*z+2.0*y*w          ],
                     [2.0*x*y + 2.0*z*w         , 1.0 - 2.0*x**2 - 2.0*z**2 , 2.0*y*z-2.0*x*w          ],
                     [2.0*x*z - 2.0*y*w         , 2.0*y*z + 2.0*x*w         , 1.0 - 2.0*x**2 - 2.0*y**2]])

def rot_x(x):
    cos_x = np.cos(x)
    sin_x = np.sin(x)
    return np.array([[1,      0,      0],
                     [0,  cos_x, -sin_x],
                     [0,  sin_x,  cos_x]])

def rot_y(y):
    cos_y = np.cos(y)
    sin_y = np.sin(y)
    return np.array([[ cos_y, 0, sin_y],
                     [     0, 1,     0],
                     [-sin_y, 0, cos_y]])

def rot_z(z):
    cos_z = np.cos(z)
    sin_z = np.sin(z)
    return np.array([[cos_z, -sin_z, 0],
                     [sin_z,  cos_z, 0],
                     [    0,      0, 1]])

def rot_rpy(r, p, y):
    return rot_z(y).dot(rot_y(p).dot(rot_x(r)))

def jac_to_np(jacobian):
    res = np.zeros((jacobian.rows(), jacobian.columns()))
    for idx_c in range(jacobian.columns()):
        col = jacobian.getColumn(idx_c)
        for idx_r in range(jacobian.rows()):
            res[idx_r, idx_c] = col[idx_r]
    return res



