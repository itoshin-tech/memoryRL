#coding:utf-8
import numpy as np 
import pickle

import pdb
import core

AGT_NAME = 'agt_q'

class Agt(core.coreAgt):
    def __init__(
        self,
        n_action=2,
        init_val_Q=0,
        epsilon=0.1,
        alpha=0.1,
        gamma=0.9,
        input_size=(7, ),
        filepath=None,
        filepath_all=None,
        max_memory=500,
    ):
        self.n_action = n_action
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.init_val_Q = init_val_Q
        self.input_size=input_size

        self.filepath = filepath
        self.filepath_all = filepath_all
        self.max_memory=max_memory

        # variables
        self.time = 0
        self.Q = {}
        self.len_Q = 0
    

    def select_action(self, observation):
        obs = self._trans_code(observation)

        self._check_and_add_observation(obs)
        
        if self.epsilon < np.random.rand():
            action = np.argmax(self.Q[obs])
        else:
            action = np.random.randint(0, self.n_action)
        return action
    
    def get_Q(self, observation):
        obs = self._trans_code(observation)
        if obs in self.Q:
            val = self.Q[obs]
            return np.array(val)
        else:
            return (None, ) * self.n_action

    def _check_and_add_observation(self, observation):
        if observation not in self.Q:
            self.Q[observation] = [self.init_val_Q] * self.n_action
            self.len_Q += 1
            if self.len_Q > self.max_memory:
                raise ValueError('Qの大きさが上限 %dに達しました。' % self.max_memory)
            else:
                if (self.len_Q < 100 and self.len_Q % 10 == 0) or (self.len_Q % 100 == 0):
                    print('used memory for Q-table --- %d' % self.len_Q)
    
    def _trans_code(self, observation):
        # obs = "_".join([str(o) for o in observation])
        obs = str(observation)
        return obs
    
    def learn(self, observation, action, reward, next_observation, done):
        """
        Q(obs, act) 
            <- (1-alpha) Q(obs, act) 
                + alpha ( rwd + gammma * max_a Q(next_obs))
        
        input : (obs, act)
        output: Q(obs, act)
        target: rwd * gamma * max_a Q(next_obs, a)
        """
        obs = self._trans_code(observation)
        next_obs = self._trans_code(next_observation)

        self._check_and_add_observation(next_obs)

        output = self.Q[obs][action]
        if done is False:
            target = reward + self.gamma * max(self.Q[next_obs])
        else:
            target = reward

        self.Q[obs][action] -= self.alpha * (output - target)
    
    def save_weights(self, filepath=None):
        if filepath is None:
            filepath = self.filepath
        with open(filepath, mode='wb') as f:
            pickle.dump(self.Q, f)

    def load_weights(self, filepath=None):
        if filepath is None:
            filepath = self.filepath
        with open(filepath, mode='rb') as f:
            self.Q = pickle.load(f)




    

    