import numpy as np
import robotics_math as rm

class Leg:
    def __init__(self):

        self.q     = np.ndarray((3,4))
        self.w     = np.ndarray((3,3))
        self.v     = np.ndarray((3,3))
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

    def pk1(self, u, v, w):
        u_prime = u - w.dot(w.T.dot(u))
        v_prime = v - w.dot(w.T.dot(v))
        numer = w.T.dot(rm.skew_3d(u_prime[:,0]).dot(v_prime))
        denom = u_prime.T.dot(v_prime)
        numer = numer[0][0]
        denom = denom[0][0]
        # print(numer[0][0])
        # print(denom[0][0])
        return np.arctan2(numer,denom)

    def pk2(self, u, v, w1, w2):
        w1 = w1.reshape(3,1)
        w2 = w2.reshape(3,1)
        w1_cross_w2 = rm.skew_3d(w1[:,0]).dot(w2.reshape(3,1))

        alpha = w1.T.dot(v)[0][0]
        beta  = w2.T.dot(u)[0][0]
        gamma_squared = np.linalg.norm(u)**2 - alpha**2 - beta**2
        gamma_squared = gamma_squared / np.linalg.norm(w1_cross_w2)**2

        z_pos = alpha * w1 + beta * w2 + np.sqrt(gamma_squared) * w1_cross_w2
        z_neg = alpha * w1 + beta * w2 - np.sqrt(gamma_squared) * w1_cross_w2

        theta1_p = self.pk1(v, z_pos, -w1)
        theta2_p = self.pk1(u, z_pos, w2)

        theta1_n = self.pk1(v, z_neg, -w1)
        theta2_n = self.pk1(u, z_neg, w2)

        return (theta1_p, theta2_p), (theta1_n, theta2_n)


    def ik(self, pos, sign=-1):
        dist = np.linalg.norm(pos)
        res = dist**2 - np.linalg.norm(self.Ls)**2
        res = res / (2 * self.Ls[1] * self.Ls[2])
        # TODO make a flag
        theta3 = sign*np.abs(np.arccos(res))

        q = self.gst0[0:4,3]
        q = np.dot(rm.prod_exp(self.twist[:,2:], [theta3]), q)


        u = q[0:3].reshape(3,1)
        v = pos.reshape(3,1)

        sol1, sol2 = self.pk2(u, v, self.w[0:3,0], self.w[0:3,1])
        theta1, theta2 = sol2

        # print(sol1)
        # print(sol2)

        # eL = np.sqrt(self.Ls[2]**2 + self.Ls[1]**2 + 2*self.Ls[1]*self.Ls[2]*np.cos(theta3))
        # phi = np.arcsin(pos[0] / eL)
        # beta_asin = self.Ls[2] * np.sin(np.pi - theta3) / eL
        # theta2 = phi + np.arcsin(beta_asin)
        # theta2 = -theta2
#
        # projL = self.Ls[1] * np.cos(theta2) + self.Ls[2] * np.cos(theta2 + theta3)
        # planeL = np.sqrt(projL**2 + self.Ls[0]**2)
        # gamma = np.arctan(self.Ls[0]/projL)
        # theta1 = np.arccos(-pos[2] / planeL) - gamma

        return np.array([theta1, theta2, theta3])
L = Leg()
print(L.twist)



print("\ntest1")
rand_joints = np.array([-0.1,0.5,-0.1])
print(rand_joints)
res = L.fk(rand_joints)
# print(res)
res = L.ik(res[0:3,3])
print(res)
res = L.fk(res)
# print(res)

print("\ntest2")
rand_joints = np.array([-0.1,-0.5,-0.1])
print(rand_joints)
res = L.fk(rand_joints)
print(res)
res = L.ik(res[0:3,3])
print(res)
res = L.fk(res)
print(res)
#
print("\ntest3")
rand_joints = np.array([0.1,-0.5,-0.1])
print(rand_joints)
res = L.fk(rand_joints)
print(res)
res = L.ik(res[0:3,3])
print(res)
res = L.fk(res)
print(res)
#
#
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
#
#
print("\nTest1")
L = Leg()
p1 = np.array([0.0,-0.146, -.40116])
p2 = np.array([0.0, 0.146, -.40116])
res = L.ik(p1)
print(res)
res = L.ik(p2)
print(res)
p3 = np.array([-0.1, 0.60023166, -0.1       ])
res = L.fk(p3)
print(res)


res = L.fk(np.array([0,0,0]))
print(res)
