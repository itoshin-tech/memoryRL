"""
core.py
EnvとAgtの抽象クラス
"""
import numpy as np


class coreEnv():
    """
    Envの抽象クラス
    """
    def __init__(self):
        pass

    def reset(self):
        """
        変数を初期化
        """

    def step(self, action):  # pylint:disable=no-self-use
        """
        action に従って、observationを更新
        """
        if action == 0:
            observation = None
            reward = None
            done = None
        else:
            observation = None
            reward = None
            done = None
        return observation, reward, done

    def render(self):  # pylint:disable=no-self-use
        """
        内部状態に対応したimg を作成
        """
        img = np.ones((100, 100, 3), type=np.uint8) * 255  # pylint:disable=unexpected-keyword-arg
        return img

    def seed(self, seed):  # pylint:disable=no-self-use
        """
        乱数のシードを指定
        """
        np.random.seed(seed)


class coreAgt():
    """
    Agtの抽象クラス
    """
    def __init__(self):
        pass

    def build_model(self):
        """
        モデル構築(Tensorflow使用時)
        """

    def select_action(self, observation):  # pylint:disable=no-self-use
        """
        observation に基づいてaction を出力
        """
        if observation == 0:
            action = None
        else:
            action = None
        return action

    def get_Q(self, observation):  # pylint:disable=no-self-use
        """
        observationに対するQ値を出力
        """
        if observation == 0:
            Q = None
        else:
            Q = None
        return Q

    def learn(self, observation, action, reward, next_observation, done):
        """
        学習
        """

    def reset(self):
        """
        内部状態をリセット(lstmやgruで使用)
        """

    def save_state(self):
        """
        内部状態をメモリーに保存(lstmやgruで使用)
        """

    def load_state(self):
        """
        内部状態の復元(lstmやgruで使用)
        """

    def save_weights(self, filepath):
        """
        Qtableやweightパラメータの保存
        """

    def load_weights(self, filepath):
        """
        Qtableやweightパラメータの読み込み
        """
    