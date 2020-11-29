"""
agt_gru.py
ニューラルネットとGRUを使ったエージェント
"""
import sys
import numpy as np
import tensorflow as tf
import replaymemory as mem
import core


class Agt(core.coreAgt):
    """
    出力層の手前にGRUを配置したエージェント
    """
    def __init__(
        self,
        n_action=2,
        input_size=(7, ),
        epsilon=0.1,
        gamma=0.9,
        n_dense=32,
        n_dense2=None,
        n_lstm=8,
        memory_size=20,
        learn_interval=10,
        data_length_for_learn=20,
        epochs_for_train=1,
        batch_size=1,
        data_length=1,
        filepath=None,
    ):
        """
        Parameters
        ----------
        n_action: int
            行動の種類の数
        input_size: tuple of int 例 (7,), (5, 5)
            入力の次元
        epsilon: float (0から1まで)
            Q学習のε、乱雑度
        gammma: float (0から1まで)
            Q学習のγ、割引率
        n_dense: int
            中間層1のニューロン数
        n_dense2: int or None
            中間層2のニューロン数
            None の場合は中間層2はなし
        n_lstm: int
            gru層のユニット数
        memory_size: int
            メモリーに蓄えるエピソード数
        learn_interval: int
            何ステップ毎に学習するか
        data_length_for_learn: int
            1回の学習時につかう連続データの長さ
        epochs_for_train: int
            1回の学習のエポック数(1でよい)
        batch_size: int (1にすること)
            バッチサイズ　今は1のみで対応
        data_length: int (1にすること)
            一度にモデルに入れるエピソード数 今は1のみで対応
        filepath: str
            セーブ用のファイル名
        """

        # パラメータ
        self.n_action = n_action
        self.input_size = input_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_dense = n_dense
        self.n_dense2 = n_dense2
        self.n_lstm = n_lstm
        self.memory_size = memory_size
        self.learn_interval = learn_interval
        self.data_length_for_learn = data_length_for_learn
        self.epochs_for_train = epochs_for_train
        self.batch_size = batch_size
        self.data_length = data_length
        self.filepath = filepath

        super().__init__()
        # 変数
        self.time = 0

    def build_model(self):
        # memory instanse生成
        self.replay_memory = mem.ReplayMemory(
            memory_size=self.memory_size,
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
        except:
            print('obs のサイズが間違っています。')
            sys.exit()

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
        obss, acts, rwds, _, dones = outs

        Xss = []
        Tss = []

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
        for dat in data:
            Xs = np.array(dat[0]).reshape((1, 1, -1))
            Ts = np.array(dat[1]).reshape((1, 1, -1))
            agt.model.fit(
                {'input': Xs},
                {'dense_out': Ts},
                batch_size=1,
                verbose=0,
                epochs=1,
                shuffle=False,
            )

    # prediction
    agt.lstm.reset_states(init_h)
    for dat in data:
        Xs = np.array(dat[0]).reshape((1, 1, -1))
        out = agt.model.predict(Xs)
        agt.state = out[1:]
        print(Xs.reshape(-1), np.round(out[0].reshape(-1),2))  # out = [output, state_h, state_c]
