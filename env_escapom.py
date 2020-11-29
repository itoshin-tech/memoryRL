# coding:utf-8
import numpy as np
import cv2
import pdb

import core

class Env(core.coreEnv):
    """
    初めだけしかゴールが表示されない
    """
    name = 'env_escapom'
    ID_brank = 0
    ID_agt = 1
    ID_goal = 3

    def __init__(
            self,
            n_action = 2,
            field_length = 5,
            range_reward = [-1, 0, 1], # fail, walk, escape
            pos_start = 0,
            goal_candidate = [2, 3],
            is_pomdp=True,
        ):
        # parameters
        self.n_action = n_action
        self.field_length = field_length
        self.range_reward = range_reward
        self.pos_start = pos_start
        self.goal_candidate = goal_candidate
        self.is_pomdp = is_pomdp

        # variables        
        self.pos_goal = None
        self.field = None
        self.time = None
        self.is_first_step = None
        self.done = None
        self.reward = None
        self.action = None

    def reset(self):
        self.done = False
        self.reward = None
        self.action = None

        self.time = 0
        self.pos_agt = self.pos_start
        self.field = np.ones(self.field_length, dtype=int) * Env.ID_brank
        idx = np.random.randint(len(self.goal_candidate))
        self.pos_goal = self.goal_candidate[idx]
        
        self.is_first_step = True
        observation = self._make_observation()
        return observation

    def step(self, action):
        # next state
        if action == 1: # walk
            next_pos = self.pos_agt + 1
            if next_pos >= self.field_length:
                reward = self.range_reward[0] # no try
                done = True
            else:
                self.pos_agt = next_pos
                reward = self.range_reward[1] # walk
                done = False
        elif action == 0: # try
            if self.pos_agt == self.pos_goal:
                reward = self.range_reward[2] # success
                done = True
            else:
                reward = self.range_reward[0] # fail
                done = True
        
        self.is_first_step = False
        observation = self._make_observation()

        # for render
        self.done = done
        self.reward = reward
        self.action = action
        
        return observation, reward, done
    
    def _make_observation(self):
        observation = self.field.copy()
        if self.is_first_step is True:
            observation[self.pos_goal] = Env.ID_goal
        if self.is_pomdp is False:
            observation[self.pos_goal] = Env.ID_goal
        observation[self.pos_agt] = Env.ID_agt
        return observation
    
    def render(self):
        # parameter 
        unit = 50
        col_brank = (0, 255, 0)
        col_agt = (255, 255, 255)
        col_agt_edge = (0, 0, 0)
        col_goal = (255, 100, 0)
        col_miss = (0, 0, 255)
        col_success = (0, 255, 0)

        width = unit * self.field_length
        height = unit

        img = np.zeros((height, width, 3), dtype=np.uint8)

        # background
        if self.reward is None:
            col = col_brank
        elif (self.reward == self.range_reward[0]) and (self.done is True):
            col = col_miss
        elif self.reward == self.range_reward[2]:
            col = col_success
        else:
            col = col_brank

        cv2.rectangle(img, (0, 0), (width-1, height-1), col, -1)

        # goal
        if (self.is_first_step is True) or (self.is_pomdp is False):
            r0 = (unit * self.pos_goal, 0)
            r1 = (unit * (self.pos_goal + 1), height - 1)
            cv2.rectangle(img, r0, r1, col_goal, -1)

        # agt
        if self.action == 0:
            radius = int(unit * 0.2)
        else:
            radius = int(unit * 0.4)
        r0 = (int(unit * self.pos_agt + unit/2), int(unit / 2))
        cv2.circle(img, r0, radius, col_agt, -1)
        cv2.circle(img, r0, radius, col_agt_edge, 2)

        return img

    def seed(self, seed):
        np.random.seed(seed)

def show_obs(obs, act, rwd, done):
    print('%s' % obs)
    if act is None:
        pass    
    else:
        print('%s act:%d, rwd:% .2f, done:%s' % (obs, act, rwd, done))

if __name__ == '__main__':
    env = Env(
        field_length=5,
        goal_candidate=[2, 3],
        )
    obs = env.reset()
    print(obs)
    is_process = False
    done = False
    while True:
        img = env.render()
        cv2.imshow('img', img)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        if key == ord('d'):
            act = 0
            is_process = True

        if key == ord('f'):
            act = 1
            is_process = True

        if is_process is True:
            if done is True:
                print('start')
                obs = env.reset()
                act = None
                rwd = None
                done = False
            else:
                obs, rwd, done = env.step(act)

            show_obs(obs, act, rwd, done)

            is_process = False
    




    





   




    


