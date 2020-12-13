"""
mng_agt_history.py
エージェントの学習履歴管理
"""
import numpy as np
import matplotlib.pyplot as plt


class MngAgtHistory:
    """
    エージェントの学習履歴を管理
    """
    def __init__(self, agt_names, pathnames):
        self.agt_names = agt_names
        self.load_history(pathnames)

    def load_history(self, pathnames):
        """
        履歴をロード
        """
        self.Qs = []  # pylint:disable=invalid-name
        self.eval_rwds = []
        self.eval_steps = []
        self.eval_x = []
        for pathname in pathnames:
            hist = np.load(pathname + '.npz')
            self.Qs.append(hist['hist_Qss'].tolist())
            self.eval_rwds.append(hist['eval_rwds'].tolist())
            self.eval_steps.append(hist['eval_steps'].tolist())
            self.eval_x.append(hist['eval_x'].tolist())

    def show_all_graphs(self, agt_names, show_Q=False):  # pylint:disable=invalid-name
        """
        グラフを描画
        """
        col = ['#f00', '#0a0', '#00f', '#0aa']
        plt.figure(figsize=(10,5))
        plt.subplots_adjust(hspace=0.6)
        plt.subplot(211)
        for i, agt_name in enumerate(self.agt_names):
            yval = self.eval_rwds[i]
            xval = self.eval_x[i]
            plt.plot(xval, yval, col[i], marker='.', label=agt_name)
        plt.title('reward / episode')
        plt.grid(True)
        plt.legend(loc='lower left')

        plt.subplot(212)
        if show_Q is True:
            for i, agt_name in enumerate(agt_names):
                Qs = np.array(self.Qs[i])
                yval1 = Qs[:, 0, 0].reshape(-1).tolist()  # 0) foward, 1)turn left
                yval2 = Qs[:, 1, 0].reshape(-1).tolist()  # 2)turn right
                xval = self.eval_x[i]
                plt.plot(xval, yval1, col[i], marker='.', linestyle='-', label=agt_name)
                plt.plot(xval, yval2, col[i], marker='.', linestyle=':')
            plt.title('Q')
        else:
            for i, agt_name in enumerate(agt_names):
                yval = self.eval_steps[i]
                xval = self.eval_x[i]
                plt.plot(xval, yval, col[i], marker='.', linestyle='-', label=agt_name)
            plt.title('steps / episode')

        plt.grid(True)
        plt.legend(loc='lower left')
        plt.show()
