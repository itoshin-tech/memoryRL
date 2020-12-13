"""
trainer.py
環境とエージェントを組み合わせて動かす
"""
import time
import cv2
import numpy as np


class Trainer:
    """
    環境でエージェントを動かす
    一定ステップ毎に評価をし記録する
    """
    def __init__(
        self,
        agt=None,
        env=None,
        eval_env=None,
        obss=None,
        is_show_Q=True,
        show_header='',
        ):
        self.agt = agt
        self.env = env
        self.eval_env = eval_env
        self.obss = obss
        self.is_show_Q = is_show_Q
        self.show_header = show_header

        # 変数
        self.hist_Qss = []
        self.hist_eval_rwds_in_episode = []
        self.hist_eval_steps_in_episode = []
        self.hist_eval_x = []
        self.hist_start_x = 0

        # 学習履歴データ保存クラスのインスタンス生成
        self.eval_simhist = SimHistory()

    def simulate(self,
        N_STEP=1000,
        n_episode=-1,
        SHOW_Q_INTERVAL=100,
        IS_LEARN=True,
        is_animation=False,
        show_delay=0.5,
        eval_N_STEP=-1,
        eval_n_episode=-1,
        eval_epsilon=0.0,
        eval_show_delay=0.1,
        eval_is_animation=False,
        EARY_STOP_STEP=None,
        EARY_STOP_REWARD=None,
        ):
        """
        エージェントを環境で動かす
        """

        stime = time.time()

        obs = self.env.reset()
        self.agt.reset()

        timestep = 0
        episode = 0
        done = False
        self.on_simulation = True

        # シミュレーション
        while self.on_simulation:
            # 通常のステップ
            if done is False:
                act = self.agt.select_action(obs)
                next_obs, rwd, next_done = self.env.step(act)

                # 学習
                if IS_LEARN is True:
                    self.agt.learn(obs, act, rwd, next_obs, next_done)

            # 最終状態での処理
            else:
                next_obs = self.env.reset()
                rwd = None
                next_done = False
                self.agt.reset()

            obs = next_obs
            done = next_done

            # アニメーション描画
            if is_animation:
                img = self.env.render()
                cv2.imshow('trainer', img)
                key = cv2.waitKey(int(show_delay * 1000))
                if key == ord('q'):
                    self.off_simulation()
                    break

            # 一定の間隔で記録と評価と途中経過表示
            if (timestep % SHOW_Q_INTERVAL == 0) and (timestep > 0):
                hist_Qs = self._show_Q() #  self.is_show_Q == True で表示
                self.hist_Qss.append(hist_Qs)

                # 評価
                eval_rwds_in_episode = None
                eval_steps_in_episode = None
                if (eval_N_STEP > 0) or (eval_n_episode > 0):
                    out = self.evaluation(
                        eval_N_STEP=eval_N_STEP,
                        eval_n_episode=eval_n_episode,
                        eval_epsilon=eval_epsilon,
                        eval_is_animation=eval_is_animation,
                        eval_show_delay=eval_show_delay,
                    )
                    eval_rwds_in_episode = out.mean_rwds[0]
                    eval_steps_in_episode = out.meaN_STEPs_in_episode[0]
                    self.hist_eval_rwds_in_episode.append(eval_rwds_in_episode)
                    self.hist_eval_steps_in_episode.append(eval_steps_in_episode)
                    self.hist_eval_x.append(self.hist_start_x + timestep)

                # 途中経過表示
                ptime = time.time() - stime
                if eval_rwds_in_episode is not None:
                    print('%s %d --- %d sec, eval_rwd % .2f, eval_steps % .2f' % (
                            self.show_header,
                            timestep, ptime,
                            eval_rwds_in_episode,
                            eval_steps_in_episode,
                            )
                        )
                else:
                    print('%s %d --- %d sec, no evaluation' % (
                            self.show_header,
                            timestep, ptime,
                            )
                        )


                # 条件をクリアしていたら途中で学習を停止
                if EARY_STOP_STEP is not None:
                    if EARY_STOP_STEP >= eval_steps_in_episode:
                        print('EARY_STOP_STEP %d >= %d' % \
                              (EARY_STOP_STEP, eval_steps_in_episode))
                        return
                if EARY_STOP_REWARD is not None:
                    if EARY_STOP_REWARD <= eval_rwds_in_episode:
                        print('EARY_STOP_REWARD %.5f <= %.5f' % \
                              (EARY_STOP_REWARD, eval_rwds_in_episode))
                        return

            # 指定したstepかepisode数に達したら終了
            if N_STEP > 0:
                if timestep >= N_STEP:
                    break
            if n_episode > 0:
                if episode > n_episode:
                    break

            timestep += 1
            episode += done

        return

    def off_simulation(self):
        """
        シミュレーションを終了する
        """
        self.on_simulation = False

    def evaluation(self,
            eval_N_STEP,
            eval_n_episode,
            eval_epsilon,
            eval_is_animation,
            eval_show_delay,
        ):
        """
        学習を止めてエージェントを環境で動作させ、
        パフォーマンスを評価する
        """
        self.agt.save_state()
        epsilon_backup = self.agt.epsilon
        self.agt.epsilon = eval_epsilon
        self.agt.reset()
        self.eval_simhist.reset()
        obs = self.eval_env.reset()

        # simulate
        timestep = 0
        episode = 0
        done = False
        while True:
            if done is False:
                act = self.agt.select_action(obs)
                next_obs, rwd, next_done = self.eval_env.step(act)
                self.eval_simhist.add(act, rwd, next_done) # 記録用
            else:
                next_obs = self.eval_env.reset()
                rwd = None
                next_done = False
                self.agt.reset()
            obs = next_obs
            done = next_done

            if eval_is_animation:
                img = self.env.render()
                cv2.imshow('eval', img)
                cv2.waitKey(int(eval_show_delay * 1000))

            if eval_N_STEP > 0:
                if timestep >= eval_N_STEP:
                    break
            if eval_n_episode > 0:
                if episode > eval_n_episode:
                    break

            timestep += 1
            episode += done
        self.eval_simhist.record()
        cv2.destroyWindow('eval')

        self.agt.load_state()
        self.agt.epsilon = epsilon_backup

        return self.eval_simhist

    def _show_Q(self):
        assert isinstance(self.obss, list)

        self.agt.save_state()
        self.agt.reset()
        Qs = []
        if self.agt.n_action == 2:
            self._log('observation       : Q(0), Q(1)')
        else:
            self._log('observation       : Q')
        for obss in self.obss:
            for obs in obss:
                if self.agt.n_action == 2:
                    val = self.agt.get_Q(np.array(obs))
                    if val[0] is None:
                        pass
                    elif val[0] > val[1]:
                        dirc = '<--'
                    else:
                        dirc = '-->'

                    if val[0] is None:
                        self._log('%s : None' % (str(np.array(obs))))
                        Qs.append((None, ) * self.agt.n_action)
                    else:
                        msg = '%s : % 5.2f %s % 5.2f' \
                            % (str(np.array(obs)), val[0], dirc, val[1])
                        self._log(msg)
                        Qs.append(val.copy())
                else:
                    val = self.agt.get_Q(np.array(obs))
                    self._log('%s' % (str(np.array(obs))))
                    if val[0] is not None:
                        self._log('Q:' + str(np.round(val, 2)))
                        Qs.append(val.copy())
                    else:
                        Qs.append((None,) * self.agt.n_action)

            self._log('')
        self.agt.load_state()
        return Qs

    def _log(self, msg):
        if self.is_show_Q is True:
            print(msg)

    def save_history(self, pathname):
        """
        履歴をセーブ
        """
        np.savez(pathname + '.npz',
            hist_Qss=self.hist_Qss,
            eval_rwds=self.hist_eval_rwds_in_episode,
            eval_steps=self.hist_eval_steps_in_episode,
            eval_x=self.hist_eval_x,
        )

    def load_history(self, pathname):
        """
        履歴をロード
        """
        hist = np.load(pathname + '.npz')
        self.hist_Qss = hist['hist_Qss'].tolist()
        self.hist_eval_rwds_in_episode = hist['eval_rwds'].tolist()
        self.hist_eval_steps_in_episode = hist['eval_steps'].tolist()
        self.hist_eval_x = hist['eval_x'].tolist()
        self.hist_start_x = self.hist_eval_x[-1]


class SimHistory:
    """
    シミュレーションの履歴保存クラス
    """
    def __init__(self):
        self.reset()

    def reset(self, init_step = 0):
        """
        履歴をリセット
        """
        self.mean_rwds = []
        self.rwds = []
        self.rwd  = 0

        self.meaN_STEPs_in_episode = []
        self.steps_in_episode = []
        self.step_in_episode = 0

        self.steps_for_mean = []
        self.stepcnt = init_step

    def add(self, act, rwd, done):  # pylint:disable=unused-argument
        """
        履歴の追加
        """
        self.rwd += rwd
        self.step_in_episode += 1
        self.stepcnt += 1
        if done is True:
            self.rwds.append(self.rwd)
            self.rwd = 0
            self.steps_in_episode.append(self.step_in_episode)
            self.step_in_episode = 0

    def record(self):
        """
        履歴の平均の計算と保存
        """
        self.mean_rwds.append(np.mean(self.rwds))
        self.rwds = []

        self.meaN_STEPs_in_episode.append(np.mean(self.steps_in_episode))
        self.steps_in_episode = []
        self.steps_for_mean.append(self.stepcnt)
