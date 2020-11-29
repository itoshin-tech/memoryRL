#coding:utf-8
import numpy as np 
import tensorflow as tf

import pdb

import core

AGT_NAME = 'adg_qnet'

class Agt(core.coreAgt):
    def __init__(
        self,
        n_action=2,
        epsilon=0.1,
        gamma=0.9,
        input_size=(7, ),
        n_dense=32,
        n_dense2=None,
        filepath=None,
        filepath_all=None,
    ):
        self.n_action = n_action
        self.epsilon = epsilon
        self.gamma = gamma
        self.input_size = input_size
        self.n_dense = n_dense
        self.n_dense2 = n_dense2
        self.filepath = filepath
        self.filepath_all = filepath_all

        # variables
        self.time = 0

    def build_model(self):
        inputs = tf.keras.Input(shape=(self.input_size))
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(self.n_dense, activation='relu')(x)

        if self.n_dense2 is not None:
            x = tf.keras.layers.Dense(self.n_dense2, activation='relu')(x)

        outputs = tf.keras.layers.Dense(self.n_action, activation='linear')(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mse']
        )

    def select_action(self, observation):
        Q = self.get_Q(observation)

        if self.epsilon < np.random.rand():
            action = np.argmax(Q)
        else:
            action = np.random.randint(0, self.n_action)
        return action
    
    def get_Q(self, observation):
        obs = self._trans_code(observation)
        Q = self.model.predict(obs.reshape((1,) + self.input_size))[0]
        return Q
    
    def _trans_code(self, observation):
        return observation

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
        Q = self.model.predict(obs.reshape((1,) + self.input_size))

        if done is False:
            next_Q = self.model.predict(next_obs.reshape((1,) + self.input_size))[0]
            target = reward + self.gamma * max(next_Q)
        else:
            target = reward

        Q[0][action] = target
        self.model.fit(obs.reshape((1,) + self.input_size), Q, verbose=0, epochs=1)

    def save_weights(self, filepath=None):
        if filepath is None:
            filepath = self.filepath
        self.model.save_weights(filepath, overwrite=True)
    
    def load_weights(self, filepath=None):
        if filepath is None:
            filepath = self.filepath
        self.model.load_weights(filepath)

if __name__ == '__main__':
    agt = Agt()
    agt.input_size = (7, 7)
    agt.n_dense = 16
    agt.n_dense2 = 16
    agt.build_model()
    agt.model.summary()



    