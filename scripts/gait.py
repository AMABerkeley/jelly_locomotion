import numpy as np
import robotics_math as rm
from leg_kin import Leg

class Gait:
    def __init__(self, phis, betas, p1_fl, p2_fl, p1_fr, p2_fr, p1_rl, p2_rl, p1_rr, p2_rr, mode=None):
        # Leg order Convention FL, FR, RL, RR
        #                      11  22  33  44
        self.phis  = phis
        self.betas = betas
        self.p1 = [p1_fl, p1_fr, p1_rl, p1_rr]
        self.p2 = [p2_fl, p2_fr, p2_rl, p2_rr]
        self._leg = Leg()

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

    def stance(self, i, rel_time):
        assert rel_time <= 1
        assert rel_time >= 0
        return self.p2[i] * rel_time**0.8 + self.p1[i] * (1 - rel_time**0.8)

    def swing(self, i, rel_time):
        assert rel_time <= 1
        assert rel_time >= 0
        self.p2 + np.cos(rel_time*np.pi/2)
        height = 0.08
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

class SimpleWalkingGait(Gait):
    def __init__(self, beta, p1, p2, mode=None):
        assert beta < 1
        assert beta >= 0.75

        phis = [0, 0.5, beta, beta - 0.5]
        betas =[beta, beta, beta, beta]
        Gait.__init__(self, phis, betas, p1, p2, p1, p2, p1, p2, p1, p2, mode=mode)
