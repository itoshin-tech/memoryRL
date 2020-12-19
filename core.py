"""
core.py
EnvとAgtの抽象クラス
"""
from abc import ABCMeta, abstractmethod


class coreEnv(metaclass=ABCMeta):
    """
    Envの抽象クラス
    """

    def reset(self):
        """
        変数を初期化
        """

    @abstractmethod
    def step(self, action: int):  # pylint:disable=no-self-use
        """
        action に従って、observationを更新

        Returns
        -------
        observation: np.ndarray
        reward: int
        done: bool
        """
        raise NotImplementedError()

    @abstractmethod
    def render(self):  # pylint:disable=no-self-use
        """
        内部状態に対応したimg を作成

        Returns
        -------
        img: np.ndarray((h, w, 3), type=np.uint8)
        """
        raise NotImplementedError()


class coreAgt(metaclass=ABCMeta):
    """
    Agtの抽象クラス
    """

    def build_model(self):
        """
        モデル構築(Tensorflow使用時)
        """

    @abstractmethod
    def select_action(self, observation):  # pylint:disable=no-self-use
        """
        observation に基づいてaction を出力

        Returns
        -------
        action: int
        """
        raise NotImplementedError()

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
    