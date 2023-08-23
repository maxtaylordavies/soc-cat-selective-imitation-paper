import numpy as np


class Gridworld:
    def __init__(self, width, height, goals, move_cost=0, gamma=0.9):
        super().__init__()
        self.width = width
        self.height = height
        self.move_cost = move_cost
        self.gamma = gamma

        self.state_dim, self.action_dim = self.height * self.width, 4

        self._reward_map = {self.state_to_idx(s): 0 for s in goals}
        self._action_map = {
            0: np.array([0, -1]),  # left
            1: np.array([-1, 0]),  # up
            2: np.array([0, 1]),  # right
            3: np.array([1, 0]),  # down
        }

        self._create_transition_matrix()

    def _create_transition_matrix(self):
        self.P = np.zeros((self.state_dim, self.action_dim, self.state_dim))

        for row in range(self.height):
            for col in range(self.width):
                s = np.array([row, col])

                for a in self._action_map.keys():
                    s_next = s + self._action_map[a]

                    if not self._is_in_grid(s_next):
                        s_next = s

                    self.P[self.state_to_idx(s), a, self.state_to_idx(s_next)] = 1.0

    def _create_reward_matrix(self):
        self.R = np.zeros((self.state_dim, self.action_dim, self.state_dim))

        for row in range(self.height):
            for col in range(self.width):
                s = np.array([row, col])

                for a in self._action_map.keys():
                    s_next = s + self._action_map[a]

                    if not self._is_in_grid(s_next):
                        s_next = s

                    self.R[
                        self.state_to_idx(s), a, self.state_to_idx(s_next)
                    ] = self._get_reward(s_next)

    def _is_in_grid(self, s):
        return 0 <= s[0] < self.height and 0 <= s[1] < self.width

    def _get_reward(self, s):
        return self._reward_map.get(self.state_to_idx(s), 0) - self.move_cost

    def state_to_idx(self, s):
        return s[0] * self.width + s[1]

    def idx_to_state(self, idx):
        return np.array([idx // self.width, idx % self.width])

    def is_terminal(self, s):
        return self.state_to_idx(s) in self._reward_map

    def act(self, s, a):
        if self.is_terminal(s):
            return s, True

        s_next = s + self._action_map[a]
        if not self._is_in_grid(s_next):
            s_next = s

        return s_next, self._get_reward(s_next), self.is_terminal(s_next)

    def set_rewards(self, rewards):
        self._reward_map = {s: r for (s, r) in zip(self._reward_map.keys(), rewards)}
        self._create_reward_matrix()

    def solve_q_values(self, theta=0.0001):
        Q, delta = np.zeros((self.state_dim, self.action_dim)), np.inf
        while delta >= theta:
            q = Q.copy()
            Q = (self.P * (self.R + (self.gamma * q.max(axis=1)))).sum(axis=2)
            delta = abs(q - Q).max()
        return Q

    @classmethod
    def random(cls, width, height, num_goals, move_cost=0, gamma=0.9):
        goals = [
            np.array([np.random.randint(height), np.random.randint(width)])
            for _ in range(num_goals)
        ]
        return cls(width, height, goals, move_cost, gamma)
