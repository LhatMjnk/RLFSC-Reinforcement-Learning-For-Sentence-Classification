import numpy as np
import pandas as pd
import torch

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

class ECommerceReviewEnv(gym.Env):    
    def __init__(self, 
                 dataloader: DataLoader,
                 tokenizer: AutoTokenizer,
                 max_length: int,
    ):
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(max_length,))
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, **kwargs):
        self.index = 0
        self.done = False
        return self._get_observation()
    
    def step(self, action):
        ...