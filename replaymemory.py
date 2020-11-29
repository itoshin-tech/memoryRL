"""
replaymemory.py
経験の記憶モジュール
"""
import random
import numpy as np


class ReplayMemory():
    """
    経験を記録する
    """
    def __init__(self, memory_size):
        self.memory_size= memory_size
        self.index = 0
        self.memory = []
        self.sub_memory = []

    def add(self, experience, done):
        """
        経験を記憶に追加する
        """
        self.sub_memory.append(experience)
        if done is True:
            self.memory.append(self.sub_memory)
            if len(self.memory) > self.memory_size:
                self.memory.pop(0)
            self.sub_memory = []

    def sample(self, data_length):
        """
        batchs は常にスタート地点から始まる
        """
        mem_size = len(self.memory)
        if mem_size == 0:
            return None

        idx = random.sample(range(mem_size), mem_size)
        cnt = 0
        batchs = []
        while True:
            batchs = batchs + self.memory[idx[cnt]]
            if len(batchs) >= data_length:
                break
            cnt = (cnt + 1) % mem_size
        batchs = batchs[:data_length]

        out = []
        for i in range(len(batchs[0])):
            out.append(np.array([bb[i] for bb in batchs]))
        return out
