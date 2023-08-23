import numpy as np

from .distributions import boltzmann1d
from .world import Gridworld


class Agent:
    def __init__(self, v: np.ndarray):
        self.v = v
        self.Q = None

    def _get_action(self, world: Gridworld, s: np.ndarray, beta: float):
        q = self.Q[world.state_to_idx(s)]
        p = boltzmann1d(q / np.sum(q), beta)
        return np.random.choice(len(p), p=p)

    def generate_trajectory(self, world: Gridworld, beta: float, start_pos: int, max_T: int):
        world.set_rewards(self.v)
        if self.Q is None:
            self.Q = world.solve_q_values()

        traj = -1 * np.ones((max_T, 2), dtype=int)

        s = start_pos
        for t in range(max_T):
            a = self._get_action(world, s, beta)
            traj[t] = [world.state_to_idx(s), a]

            s, _, terminal = world.act(s, a)
            if terminal:
                break

        return traj[: t + 1]

    def follow_trajectory(self, world: Gridworld, traj: np.ndarray, gamma: float):
        world.set_rewards(self.v)
        s, ret = world.idx_to_state(traj[0, 0]), 0

        for t in range(len(traj)):
            a = traj[t, 1]
            s, r, terminal = world.act(s, a)

            ret += (gamma**t) * r
            if terminal:
                break

        return ret

    def reset(self):
        self.Q = None
