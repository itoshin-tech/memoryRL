#coding:utf-8

import numpy as np


class coreEnv():
    def __init__(self):
        pass

    def reset(self):
        observation = None
        return observation

    def step(self, action):
        observation = None
        reward = None
        done = None
        return observation, reward, done

    def render(self):
        img = np.ones((100, 100, 3), type=np.uint8) * 255
        return img
    
    def seed(self, seed):
        np.random.seed(seed)
    

class coreAgt():
    def __init__(self):
        pass

    def build_model(self):
        pass

    def select_action(self, observation):
        action = None
        return action

    def get_Q(self, observatin):
        Q = None
        return Q
    
    def learn(self, observation, action, reward, next_observation, done):
        pass

    def reset(self):
        pass

    def save_state(self):
        pass

    def load_state(self):
        pass

    def save_weight(self):
        pass

    def load_weight(self):
        pass

    
    