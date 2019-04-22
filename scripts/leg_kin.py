import numpy as np
import robotics_math as rm


class Leg:
    def __init__(self):

        self.q = np.ndarray((3,4))
        self.w = np.ndarray((3,3))
        self.v = np.ndarray((3,3))
        self.twist = np.ndarray((6,3))

        Ls = np.array([0.046, 0.25, 0.25116])

        self.q[0:3,0] = [0,    0, 0]
        self.q[0:3,1] = [0, Ls[0], 0]
        self.q[0:3,2] = [0, Ls[0], -Ls[1]]
        self.q[0:3,3] = [0, Ls[0], -Ls[1] - Ls[2]]

        self.Ls = Ls
        self.w[0:3,0] = [1, 0, 0]
        self.w[0:3,1] = [0, 1, 0]
        self.w[0:3,2] = [0, 1, 0]


        for i in range(0,self.w.shape[1]):
            self.v[0:3,i:i+1] = -rm.skew_3d(self.w[0:3,i]).dot(self.q[0:3,i].reshape(3,1))

        self.twist[0:3,0:3] = self.v
        self.twist[3:6,0:3] = self.w

        self.gst0 = np.eye(4)
        self.gst0[0:3,3] = self.q[0:3,3]

    def fk(self, theta):
        return np.dot(rm.prod_exp(self.twist, theta), self.gst0)

    def ik(self, pos, sign=-1):
        dist = np.linalg.norm(pos)
        res = dist**2 - np.linalg.norm(self.Ls)**2
        res = res / (2 * self.Ls[1] * self.Ls[2])
        # TODO make a flag
        theta3 = sign*np.abs(np.arccos(res))

        eL = np.sqrt(self.Ls[2]**2 + self.Ls[1]**2 + 2*self.Ls[1]*self.Ls[2]*np.cos(theta3))
        phi = np.arcsin(pos[0] / eL)
        beta_asin = self.Ls[2] * np.sin(np.pi - theta3) / eL
        theta2 = phi + np.arcsin(beta_asin)
        theta2 = -theta2

        projL = self.Ls[1] * np.cos(theta2) + self.Ls[2] * np.cos(theta2 + theta3)
        planeL = np.sqrt(projL**2 + self.Ls[0]**2)
        gamma = np.arctan(self.Ls[0]/projL)
        theta1 = np.arccos(-pos[2] / planeL) - gamma
        # theta1 = -gamma #np.arccos(-pos[2] / projL) - gamma

        return np.array([theta1, theta2, theta3])



print("\ntest1")
rand_joints = np.array([-0.1,0.5,0.1])
print(rand_joints)
L = Leg()
res = L.fk(rand_joints)
# print(res)
res = L.ik(res[0:3,3])
print(res)
res = L.fk(res)
# print(res)

print("\ntest2")
rand_joints = np.array([-0.1,-0.5,0.1])
print(rand_joints)
L = Leg()
res = L.fk(rand_joints)
# print(res)
res = L.ik(res[0:3,3])
print(res)
res = L.fk(res)
# print(res)

print("\ntest3")
rand_joints = np.array([0.1,-0.5,0.1])
print(rand_joints)
L = Leg()
res = L.fk(rand_joints)
# print(res)
res = L.ik(res[0:3,3])
print(res)
res = L.fk(res)
# print(res)


print("\nTest")
print("Leg Stand")
rand_joints = np.array([0.0, 0.6, -1.2])
print(rand_joints)
L = Leg()
res = L.fk(rand_joints)
print(res)
res = L.ik(res[0:3,3])
print(res)
res = L.fk(res)
print(res)

