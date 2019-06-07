import numpy as np
import robotics_math as rm
from leg_kin import Leg

class Gait:
    def __init__(self, phis, betas, p1_fl, p2_fl, p1_fr, p2_fr, p1_rl, p2_rl, p1_rr, p2_rr, mode=None, height=0.08):
        # Leg order Convention FL, FR, RL, RR
        #                      11  22  33  44
        self.phis  = phis
        self.betas = betas
        self.p1 = [p1_fl, p1_fr, p1_rl, p1_rr]
        self.p2 = [p2_fl, p2_fr, p2_rl, p2_rr]
        self._leg = Leg()
        self.height = height

        self.front_sign = -1
        self.rear_sign  = -1

        if mode == "crab":
            self.front_sign = 1
        if mode == "reverse_crab":
            self.rear_sign  = 1

    def _is_stance_and_time(self, idx, time_idx):
        phi  = self.phis[idx]
        beta = self.betas[idx]
        if phi + beta > 1.0:

            swing_start = (phi+beta)%1.0
            if time_idx >= swing_start and time_idx < phi:
                stance_bool = False
                rel_time = (time_idx - swing_start) / (1.0 - beta)
            else:
                if time_idx < swing_start:
                    time_slice = (1.0 - phi) + time_idx
                else:
                    time_slice = time_idx - phi
                stance_bool = True
                rel_time = time_slice / beta
        else:
            if time_idx >= phi and time_idx < beta + phi:
                stance_bool = True
                rel_time = (time_idx-phi)/ beta
            else:
                if time_idx < phi:
                    time_slice = time_idx + 1 - (phi + beta)
                else:
                    time_slice = time_idx - beta - phi
                stance_bool = False
                rel_time = time_slice / (1.0 - beta)
        return stance_bool, rel_time

    def check_stance_swing(self, time_idx):
        check_stance = []
        for i in range(4):
            check_stance.append(_is_stance_and_time(i, time_idx))
        return check_stance

    def stance(self, i, rel_time):
        assert rel_time <= 1
        assert rel_time >= 0
        return self.p2[i] * rel_time**0.8 + self.p1[i] * (1 - rel_time**0.8)

    def swing(self, i, rel_time):
        assert rel_time <= 1
        assert rel_time >= 0
        height = self.height
        return self.p1[i] * rel_time + self.p2[i] * (1 - rel_time) + np.array([0, 0, height * np.sin(rel_time * np.pi)])

    def step(self, time_idx):
        assert time_idx <= 1
        assert time_idx >=0
        positions = []
        for i in range(4):
            is_stance, rel_time = self._is_stance_and_time(i, time_idx)
            if is_stance:
                positions.append(self.stance(i, rel_time))
            else:
                positions.append(self.swing(i, rel_time))

        joints = []
        for i, po in enumerate(positions):
            if i < 2:
                joints.append(self._leg.ik(po, sign=self.front_sign))
            else:
                joints.append(self._leg.ik(po, sign=self.rear_sign))

        return np.hstack(joints)

    def set_height(self, h):
        self.height = h

class SimpleSideGait(Gait):
    def __init__(self, beta, p1, p2, mode=None, height=0.08):
        assert beta < 1
        assert beta >= 0.75
        phis = [0, beta - 0.5, 0.5, beta]
        betas =[beta, beta, beta, beta]
        lp1 = p1.copy()
        lp1[1] = -p1[1]
        lp2 = p2.copy()
        lp2[1] = -p2[1]

        rp1 = p1.copy()
        rp2 = p2.copy()

        Gait.__init__(self, phis, betas, rp1, rp2, lp1, lp2, rp1, rp2, lp1, lp2, mode=mode, height=height)

class SimpleWalkingGait(Gait):
    def __init__(self, beta, p1, p2, mode=None, height=0.08):
        assert beta < 1
        assert beta >= 0.75

        phis = [0, 0.5, beta, beta - 0.5]
        betas =[beta, beta, beta, beta]
        Gait.__init__(self, phis, betas, p1, p2, p1, p2, p1, p2, p1, p2, mode=mode, height=height)

class SimpleWalkingGaitEllipse(Gait):
    def __init__(self, beta, p1, p2, mode=None, height=0.08):
        assert beta < 1
        assert beta >= 0.75

        phis = [0, 0.5, beta, beta - 0.5]
        betas =[beta, beta, beta, beta]
        Gait.__init__(self, phis, betas, p1, p2, p1, p2, p1, p2, p1, p2, mode=mode, height=height)

    def stance(self, i, rel_time):
        assert rel_time <= 1
        assert rel_time >= 0
        height = self.height / 10.0
        return self.p2[i] * rel_time + self.p1[i] * (1 - rel_time) + np.array([0, 0, height * np.sin(rel_time * np.pi)])

    def swing(self, i, rel_time):
        assert rel_time <= 1
        assert rel_time >= 0
        height = self.height
        return self.p1[i] * rel_time + self.p2[i] * (1 - rel_time) + np.array([0, 0, height * np.sin(rel_time * np.pi)])

class SimpleMirrorWalkingGait(Gait):
    def __init__(self, beta, p1, p2, mode=None, height=0.08):
        assert beta < 1
        assert beta >= 0.75

        phis = [0, 0.5, beta, beta - 0.5]
        betas =[beta, beta, beta, beta]
        p1_r = p1.copy()
        p2_r = p2.copy()
        p1_r[0] = -p1_r[0]
        p2_r[0] = -p2_r[0]
        Gait.__init__(self, phis, betas, p1, p2, p1, p2, p2_r, p1_r, p2_r, p1_r, mode=mode, height=height)

class Jump(Gait):
    def __init__(self, beta, h, p1, p2, mode=None, height=0.08):
        assert beta < 1
        assert beta >= 0.75
        self.set_height(h)

        phis = [0, 0, 0, 0]
        betas =[beta, beta, beta, beta]
        Gait.__init__(self, phis, betas, p1, p2, p1, p2, p1, p2, p1, p2, mode=mode, height=height)

class TurningGait(Gait):
    def __init__(self, p1f, p2f, p1r, p2r, mode=None, height=0.08):
        beta = 0.8

        phis = [0, beta - 0.5, beta, 0.5]
        betas =[beta, beta, beta, beta]
        neg_mult_left = np.array([1,-1,1])

        # print("qwerqewrqewr")
        # print(p1f*neg_mult_left)
        Gait.__init__(self, phis, betas, p1f, p2f, neg_mult_left*p1f, neg_mult_left*p2f, p1r, p2r, neg_mult_left*p1r, neg_mult_left*p2r, mode=mode, height=height)


class TrotGait(Gait):
    def __init__(self, p1, p2, mode=None, height=0.08):
        beta = 0.5
        phis = [0, 0.5, 0.5, 0]
        betas =[beta, beta, beta, beta]
        Gait.__init__(self, phis, betas, p1, p2, p1, p2, p1, p2, p1, p2, mode=mode, height=height)

class BoundGait(Gait):
    def __init__(self, p1, p2, mode=None, height=0.08):
        beta = 0.5
        phis = [0.5, 0.5, 0.0, 0.0]
        betas =[beta, beta, beta, beta]
        Gait.__init__(self, phis, betas, p1, p2, p1, p2, p1, p2, p1, p2, mode=mode, height=height)
        for i in range(len(self.p1)):
            if i < 2:
                self.p1[i] += np.array([-0.05,0 , -0.05])
                self.p2[i] += np.array([-0.05,0,  -0.05])
            else:
                self.p1[i] += np.array([0.04,0 ,0.05])
                self.p2[i] += np.array([-0.04,0 ,0.05])

    def swing(self, i, rel_time):
        assert rel_time <= 1
        assert rel_time >= 0
        self.p2 + np.cos(rel_time*np.pi/2)
        height = self.height
        if i > 1:
            height += 0.1

        return self.p1[i] * rel_time + self.p2[i] * (1 - rel_time) + np.array([0, 0, height * np.sin(rel_time * np.pi)])

    def stance(self, i, rel_time):
        assert rel_time <= 1
        assert rel_time >= 0
        height = self.height
        return self.p2[i] * rel_time + self.p1[i] * (1 - rel_time) + np.array([0, 0, -height * np.sin(rel_time * np.pi)])
