#coding:utf-8
import numpy as np 
import random
import tensorflow as tf
import pdb

import replaymemory as mem
import core


AGT_NAME = 'agt_gru'

class Agt(core.coreAgt):
    def __init__(
        self,
        n_action=2,
        epsilon=0.1,
        gamma=0.9,
        input_size=(7, ),
        n_dense=32,
        n_dense2=None,
        n_lstm=8,
        memory_size=20,
        batch_size=1, # 今は1
        learn_interval=10, # 何ステップに1回fit するか
        epochs_for_train=1, # 1回の学習回数
        data_length=1, # 今は1　一度にモデルに入れるエピソード数
        data_length_for_learn=20, # 1回の学習時に何ステップ分学習させるか
        filepath=None,
        filepath_all=None,
    ):
        self.n_action = n_action
        self.epsilon = epsilon
        self.gamma = gamma
        self.input_size = input_size
        self.n_dense = n_dense
        self.n_dense2 = n_dense2
        self.n_lstm = n_lstm
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.learn_interval = learn_interval
        self.epochs_for_train = epochs_for_train
        self.data_length = data_length
        self.data_length_for_learn = data_length_for_learn
        self.filepath = filepath
        self.filepath_all = filepath_all
        
        # variables
        self.time = 0

    def build_model(self):
        # memory instanse生成
        self.replay_memory = mem.ReplayMemory(
            memory_size=self.memory_size, 
            data_length=self.data_length_for_learn,
            )
        
        # modelのinstanse生成
        self.model = self._build_model()
        self.model.compile(
            optimizer='adam',
            loss={'dense_out': 'mean_squared_error'},
            metrics=['mse']
            )
        self.lstm = self.model.get_layer('lstm')
    
    def _build_model(self):
        inputs = tf.keras.Input(
            batch_shape=((self.batch_size, self.data_length) + self.input_size),
            name='input',
            )
        # (batch_size, data_length, input_size[0], input_size[1])
        # batch_size = 1, data_length = 1

        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Flatten(),
            )(inputs)
        # (batch_size, data_length, input_size[0] * input_size[1])

        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.n_dense, activation='relu'),
            name='dense',
            )(x)
        # (batch_size, data_size, n_dense)

        if self.n_dense2 is not None:
            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.n_dense2, activation='relu'),
                name='dense2',
                )(x)
        # (batch_size, data_size, n_dense2)

        x, state_c = tf.keras.layers.GRU(
            self.n_lstm,
            return_state=True, # 内部状態を出力
            return_sequences=True, # 逐次出力
            stateful=True, # バッチ間の状態維持
            name='lstm',
            )(x)
        # (batch_size, data_length, n_lstm)
        # state_c = (1, n_lstm)

        outputs = tf.keras.layers.Dense(
            self.n_action, 
            activation='linear',
            name='dense_out',
            )(x)
        # (batch_size, data_length, n_action)

        model = tf.keras.Model(
            inputs=inputs, 
            outputs=[outputs, state_c]
            )
        
        return model

    def select_action(self, observation):
        Q = self.get_Q(observation)

        if self.epsilon < np.random.rand():
            action = np.argmax(Q)
        else:
            action = np.random.randint(0, self.n_action)
        return action
    
    def get_Q(self, observation):
        obs = self._trans_code(observation)
        try:
            Q, hstate = self.model.predict(obs.reshape((1, 1) + self.input_size))
            self.state = hstate
        except ValueError as e:
            print('obs のサイズが間違っています。')
            raise(e)

        Q = Q.reshape(-1)
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
        target: rwd + gamma * max_a Q(next_obs, a)
        """
        self.replay_memory.add((observation, action, reward, next_observation, done), done)

        if self.time % self.learn_interval ==0 and self.time > 0:
            self.save_state()
            self._fit()
            self.load_state()

        self.time += 1
    
    def reset(self):
        init_h = [np.zeros((1, self.n_lstm))]
        self.state = init_h
        self.lstm.reset_states(init_h)
    
    def save_state(self):
        self.stock_state = self.state
    
    def load_state(self):
        self.lstm.reset_states(self.stock_state)

    def _fit(self):
        sum_mem = sum([len(mm) for mm in self.replay_memory.memory])
        if sum_mem < self.data_length_for_learn:
            return
        
        self.reset()
        outs = self.replay_memory.sample(data_length=self.data_length_for_learn)
        obss, acts, rwds, next_obss, dones = outs
        is_first_step = True

        Xss = []
        Tss = []
        Tss_test = []

        # 学習データ作成
        Qss = []
        self.reset()
        for i in range(self.data_length_for_learn):
            obs = self._trans_code(obss[i])
            obs = obs.reshape((1, 1) + (self.input_size))
            Qss.append(self.model.predict(obs)[0][0][0])
            if bool(dones[i]) is True:
                self.reset()
        
        for i in range(self.data_length_for_learn - 1):
            Qs = Qss[i]
            next_Qs = Qss[i + 1]
            iact = np.argmax(next_Qs)
            if bool(dones[i]) is False:
                target = rwds[i] + self.gamma * next_Qs[iact]
            else:
                target = rwds[i]
            Qs[acts[i]] = target
            obs = self._trans_code(obss[i])
            Xss.append(obs.reshape((1, 1) + (self.input_size)))
            Tss.append(Qs.reshape(1, 1, -1))

        # 学習
        self.reset()
        for i in range(self.data_length_for_learn - 1):
            # Qs
            self.model.fit(
                {'input': Xss[i]}, 
                {'dense_out': Tss[i]}, 
                batch_size=1, 
                verbose=0, 
                epochs=self.epochs_for_train,
                shuffle=False,
            )
            if dones[i] is True:
                self.reset()

        
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
    agt.n_action = 2
    agt.n_dense = 32
    agt.n_dense2 = 32
    agt.n_lstm = 32
    agt.build_model()
    agt.model.summary()

    init_h = [np.zeros((1, agt.n_lstm))]

    # learning
    data = [
        [[1, 2, 0, 0, 0, 3, 1], [0, 0]],
        [[1, 0, 2, 0, 0, 3, 1], [0, 1]],
        [[1, 0, 0, 2, 0, 3, 1], [0, 1]],
        [[1, 0, 0, 0, 2, 3, 1], [0, 1]],
        [[1, 0, 0, 0, 0, 2, 1], [0, 1]],
        [[1, 0, 0, 0, 2, 3, 1], [1, 0]],
        [[1, 0, 0, 2, 0, 3, 1], [1, 0]],
        [[1, 0, 2, 0, 0, 3, 1], [1, 0]],
        [[1, 2, 0, 0, 0, 3, 1], [1, 0]],
    ]

    for tt in range(100):
        if tt % 10 == 0:
            print(tt)
        agt.reset()
        for t in range(len(data)):
            Xs = np.array(data[t][0]).reshape(1, 1, -1)
            Ts = np.array(data[t][1]).reshape(1, 1, -1)
            agt.model.fit(
                {'input': Xs}, # input_1, dense_1はどこで定義される？
                {'dense_out': Ts}, 
                batch_size=1, 
                verbose=0, 
                epochs=1,
                shuffle=False,
            )

    # prediction
    agt.lstm.reset_states(init_h)
    for t in range(len(data)):
        Xs = np.array(data[t][0]).reshape(1, 1, -1)
        out = agt.model.predict(Xs)
        agt.state = out[1:]
        print(Xs.reshape(-1), np.round(out[0].reshape(-1),2)) # out = [output, state_h, state_c]
    






    