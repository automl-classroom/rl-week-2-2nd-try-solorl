# from __future__ import annotations

# import gymnasium as gym


# # ------------- TODO: Implement the following environment -------------
# class MyEnv(gym.Env):
#     """
#     Simple 2-state, 2-action environment with deterministic transitions.

#     Actions
#     -------
#     Discrete(2):
#     - 0: move to state 0
#     - 1: move to state 1

#     Observations
#     ------------
#     Discrete(2): the current state (0 or 1)

#     Reward
#     ------
#     Equal to the action taken.

#     Start/Reset State
#     -----------------
#     Always starts in state 0.
#     """

#     metadata = {"render_modes": ["human"]}

#     def __init__(self):
#         """Initializes the observation and action space for the environment."""
#         pass


# class PartialObsWrapper(gym.Wrapper):
#     """Wrapper that makes the underlying env partially observable by injecting
#     observation noise: with probability `noise`, the true state is replaced by
#     a random (incorrect) observation.

#     Parameters
#     ----------
#     env : gym.Env
#         The fully observable base environment.
#     noise : float, default=0.1
#         Probability in [0,1] of seeing a random wrong observation instead
#         of the true one.
#     seed : int | None, default=None
#         Optional RNG seed for reproducibility.
#     """

#     metadata = {"render_modes": ["human"]}

#     def __init__(self, env: gym.Env, noise: float = 0.1, seed: int | None = None):
#         pass

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np

# Basically the same structure as MarsRover but with adjusted dynamics


class MyEnv(gym.Env):
    """Simple 2-state, 2-action environment with deterministic transitions.

    Actions
    -------
    Discrete(2):
    - 0: move to state 0
    - 1: move to state 1

    Observations
    ------------
    Discrete(2): the current state (0 or 1)

    Reward
    ------
    Equal to the action taken.

    Start/Reset State
    -----------------
    Always starts in state 0.
    """

    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------
    def __init__(self, horizon: int = 20, seed: int | None = None):
        super().__init__()
        self.horizon = int(horizon)
        self.current_steps = 0
        self.state = 0  # always start in state 0
        self.rng = np.random.default_rng(seed)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)

        # Convenience arrays for algorithms/tests
        self.states = np.arange(self.observation_space.n)
        self.actions = np.arange(self.action_space.n)
        self.transition_matrix = self.get_transition_matrix()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_steps = 0
        self.state = 0
        return self.state, {}

    def step(
        self, action: int
    ) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]:
        action = int(action)
        if not self.action_space.contains(action):
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")

        self.current_steps += 1

        # Deterministic transition: the next state equals the action.
        self.state = action

        reward = float(action)
        terminated = False  # no true terminal states
        truncated = self.current_steps >= self.horizon
        return self.state, reward, terminated, truncated, {}

    def get_reward_per_action(self) -> np.ndarray:
        """Return the reward matrix R[s, a]."""
        nS, nA = self.observation_space.n, self.action_space.n
        R = np.zeros((nS, nA), dtype=float)
        for a in range(nA):
            R[:, a] = float(a)
        return R

    def get_transition_matrix(self) -> np.ndarray:
        """Return the deterministic transition tensor T[s, a, s']"""
        nS, nA = self.observation_space.n, self.action_space.n
        T = np.zeros((nS, nA, nS), dtype=float)
        for a in range(nA):
            T[:, a, a] = 1.0
        return T

    # ------------------------------------------------------------------
    def render(self, mode: str = "human") -> None:
        if mode != "human":
            raise NotImplementedError("Only human render mode is supported.")
        print(f"[MyEnv] state={self.state}, step={self.current_steps}/{self.horizon}")


class PartialObsWrapper(gym.Wrapper):
    """Wrapper that makes the underlying env partially observable by injecting
    observation noise: with probability `noise`, the true state is replaced by
    a random (incorrect) observation.

    Parameters
    ----------
    env : gym.Env
        The fully observable base environment.
    noise : float, default=0.1
        Probability in [0,1] of seeing a random wrong observation instead
        of the true one.
    seed : int | None, default=None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: gym.Env, noise: float = 0.1, seed: int | None = None):
        assert 0.0 <= noise <= 1.0, "`noise` must be in [0, 1]"
        super().__init__(env)
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def _maybe_corrupt(self, true_obs: int) -> int:
        """Return the true observation or a corrupted one with prob noise."""
        if self.rng.random() < self.noise:
            return int(1 - true_obs)  # the single incorrect alternative
        return int(true_obs)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._maybe_corrupt(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self._maybe_corrupt(obs), reward, term, trunc, info

    def render(self, mode: str = "human") -> None:
        if mode != "human":
            raise NotImplementedError
        print(f"[PartialObs noise={self.noise:.2f}]", end=" ")
        self.env.render(mode=mode)
