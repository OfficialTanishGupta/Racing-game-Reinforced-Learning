import numpy as np
import gymnasium as gym
from gymnasium import spaces

from track import TRACKS

MAX_SPEED = 8.0
ACCELERATION = 0.25
FRICTION = 0.02
TURN_RATE = 0.07

RAY_ANGLES = np.radians([-90, -45, -20, 0, 20, 45, 90])
RAY_RANGE = 250.0
RAY_STEP = 6.0

PROGRESS_SCALE = 10.0
TIME_PENALTY = 0.01
CRASH_PENALTY = -10.0
BACKWARD_PENALTY = -5.0
LAP_BONUS = 20.0


class RacingEnv(gym.Env):
    def __init__(self, track=None, max_steps=4000, race_mode=False):
        super().__init__()
        self.track_name = track
        self.max_steps = max_steps
        self.race_mode = race_mode
        self.ray_distances = np.arange(RAY_STEP, RAY_RANGE + 1, RAY_STEP)
        self.observation_space = spaces.Box(-1.0, 1.0, (len(RAY_ANGLES) + 2,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
