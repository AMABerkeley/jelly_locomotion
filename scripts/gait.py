import numpy as np
import robotics_math as rm


class Gait:
    def __init__(self, phis, betas, p1, p2):
        self.phis  = phis
        self.betas = betas
        self.p1 = p1
        self.p2 = p2

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

    def stance(self, rel_time):
        assert rel_time <= 1
        assert rel_time >= 0
        return self.p2 * rel_time**0.8 + self.p1 * (1 - rel_time**0.8)

    def swing(self, rel_time):
        assert rel_time <= 1
        assert rel_time >= 0
        distance = np.linalg.norm(self.p1 - self.p2)
        self.p2 + np.cos(rel_time*np.pi/2)
        height = 0.08
        return self.p1 * rel_time + self.p2 * (1 - rel_time) + np.array([0, 0, height * np.sin(rel_time * np.pi)])

    def step(self, time_idx):
        assert time_idx <= 1
        assert time_idx >=0
        positions = []
        for i in range(4):
            is_stance, rel_time = self._is_stance_and_time(i, time_idx)
            if is_stance:
                positions.append(self.stance(rel_time))
            else:
                positions.append(self.swing(rel_time))
        return positions

class WalkingGait(Gait):
    def __init__(self, beta, p1, p2):
        assert beta < 1
        assert beta >= 0.75

        phis = [0, 0.5, beta, beta - 0.5]
        betas =[beta, beta, beta, beta]
        Gait.__init__(self, phis, betas, p1, p2)
