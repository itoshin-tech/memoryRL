#coding:utf-8

import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import os
import pdb


class Trainer:
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

        # parameters
        self.hist_Qss = []
        self.hist_eval_rwds_in_episode = []
        self.hist_eval_steps_in_episode = []
        self.hist_eval_x = []
        self.hist_start_x = 0

        # instance
        self.simhist = SimHistory()
        self.eval_simhist = SimHistory()

    def reset_history(self, init_step = 0):
        self.simhist.reset(init_step=init_step)
        self.hist_Qss = []
        self.hist_eval_rwds_in_episode = []
        self.hist_eval_steps_in_episode = []
        self.hist_eval_x = []
        self.simhist.mean_steps_in_episode = []
        self.simhist.mean_rwds = []
        self.simhist.mean_acts = []
        self.simhist.steps_for_mean = []
        self.simhist.stepcnt = init_step

    def simulate(self,
        n_step=1000,
        n_episode=-1,
        show_Q_interval=100,
        is_learn=True, 
        is_animation=False, 
        show_delay=0.5,
        eval_n_step=-1,
        eval_n_episode=-1,
        eval_epsilon=0.0,
        eval_show_delay=0.1,
        eval_is_animation=False,
        eary_stop_step=None,
        eary_stop_reward=None,
        ):

        stime = time.time()
        self.simhist.stime = time.time()

        obs = self.env.reset()
        self.agt.reset()

        timestep = 0
        episode = 0
        done = False

        # シミュレーション
        while True:
            # 通常のステップ
            if done is False:
                act = self.agt.select_action(obs)
                next_obs, rwd, next_done = self.env.step(act)

                self.simhist.add(act, rwd, next_done) # 記録用

                # 学習
                if is_learn is True:
                    self.agt.learn(obs, act, rwd, next_obs, next_done)
            
            # 最終ステップ
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
                cv2.imshow('img', img)
                cv2.waitKey(int(show_delay * 1000))

            # 一定の間隔で記録と評価と途中経過表示
            if (timestep % show_Q_interval == 0) and (timestep > 0):
                hist_Qs = self._show_Q() #  self.is_show_Q == True で表示
                self.hist_Qss.append(hist_Qs)
                self.simhist.record() # 記録

                # 評価
                eval_rwds_in_episode = -99
                eval_steps_in_episode = -99
                if (eval_n_step > 0) or (eval_n_episode > 0):
                    out = self.evaluation(
                        eval_n_step=eval_n_step,
                        eval_n_episode=eval_n_episode,
                        eval_epsilon=eval_epsilon,
                        eval_is_animation=eval_is_animation,
                        eval_show_delay=eval_show_delay,
                    )
                    eval_rwds_in_episode = out.mean_rwds[0]
                    eval_steps_in_episode = out.mean_steps_in_episode[0]
                    self.hist_eval_rwds_in_episode.append(eval_rwds_in_episode)
                    self.hist_eval_steps_in_episode.append(eval_steps_in_episode)
                    self.hist_eval_x.append(self.hist_start_x + timestep)
                # 途中経過表示
                ptime = time.time() - stime
                print('%s %d --- %d sec, rwd % .2f, eval_rwd % .2f, steps_in_e % .2f' % (
                        self.show_header,
                        timestep, ptime,
                        self.simhist.mean_rwds[-1],
                        eval_rwds_in_episode,
                        eval_steps_in_episode,
                        )
                    )
                # eary stop
                if eary_stop_step is not None:
                    if eary_stop_step >= eval_steps_in_episode:
                        print('eary_stop_step', eval_steps_in_episode)
                        return
                if eary_stop_reward is not None:
                    if eary_stop_reward <= eval_rwds_in_episode:
                        print('eary_stop_reward', eval_rwds_in_episode)
                        return
            
            # normal step
            if n_step > 0:
                if timestep >= n_step:
                    break
            if n_episode > 0:
                if episode > n_episode:
                    break

            timestep += 1
            episode += done
            
        return
    
    def evaluation(self,
            eval_n_step,
            eval_n_episode,
            eval_epsilon,
            eval_is_animation,
            eval_show_delay,
        ):
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

            if eval_n_step > 0:
                if timestep >= eval_n_step:
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
        assert type(self.obss) is list

        self._log('observation     : Q(0), Q(1)')
        self.agt.save_state()
        self.agt.reset()
        Qs = []
        for obss in self.obss:
            for obs in obss:
                if hasattr(self.agt, 'input_type'):
                    if self.agt.input_type == 'prev_act':
                        # prev_act type
                        """
                        obs = [act, obs]
                        """
                        act = obs[0]
                        obs = np.array(obs[1])
                        self.agt.prev_act = act
                        val = self.agt.get_Q(obs)
                        if act is None:
                            self._log('act = None')
                        else:
                            self._log('act = %d' % act)

                        self._log('%s' % (str(obs)))
                        self._log('Q:', np.round(val, 2))
                        Qs.append(val.copy())
                    else:
                        pdb.set_trace()
                        raise ValueError('agt.input_type が不適切。')

                elif self.agt.n_action == 2:
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
                        self._log('%s : % 5.2f %s % 5.2f' % (str(np.array(obs)), val[0], dirc, val[1]))
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
        return
    
    def save_history(self, pathname):
        np.savez(pathname + '.npz',
            mean_steps_in_episode=self.simhist.mean_steps_in_episode,
            mean_rwds=self.simhist.mean_rwds,
            mean_acts=self.simhist.mean_acts,
            hist_Qss=self.hist_Qss,
            eval_rwds=self.hist_eval_rwds_in_episode,
            eval_steps=self.hist_eval_steps_in_episode,
            eval_x=self.hist_eval_x,
            steps_for_mean=self.simhist.steps_for_mean,
            times=self.simhist.times,
        )
    
    def load_history(self, pathname):
        hist = np.load(pathname + '.npz')
        self.simhist.mean_steps_in_episode = hist['mean_steps_in_episode'].tolist()
        self.simhist.mean_rwds = hist['mean_rwds'].tolist()
        self.simhist.mean_acts = hist['mean_acts'].tolist()
        self.simhist.steps_for_mean = hist['steps_for_mean'].tolist()
        self.simhist.stepcnt = self.simhist.steps_for_mean[-1]
        self.simhist.times = hist['times'].tolist()
        self.simhist.start_time = self.simhist.times[-1]

        self.hist_Qss = hist['hist_Qss'].tolist()
        self.hist_eval_rwds_in_episode = hist['eval_rwds'].tolist()
        self.hist_eval_steps_in_episode = hist['eval_steps'].tolist()
        self.hist_eval_x = hist['eval_x'].tolist()
        self.hist_start_x = self.hist_eval_x[-1]

    
    def del_all_history(self, all_pathname):
        if os.path.isfile(all_pathname + '.npz') is True:
            os.remove(all_pathname + '.npz')


    def add_all_history(self, pathname, all_pathname):
        if os.path.isfile(all_pathname + '.npz') is True:
            self._load_all_history(all_pathname)
        else:
            print('all_history を初期化します')
            self._init_all_history()
        hist = np.load(pathname + '.npz')
        self.all_steps_in_episode.append(hist['mean_steps_in_episode'].tolist())
        self.all_rwds.append(hist['mean_rwds'].tolist())
        self.all_acts.append(hist['mean_acts'].tolist())
        self.all_Qss.append(hist['hist_Qss'].tolist())
        self.all_eval_rwds_in_episode.append(hist['eval_rwds'].tolist())
        self.all_eval_steps_in_episode.append(hist['eval_steps'].tolist())
        self.all_eval_x.append(hist['eval_x'].tolist())
        self.all_steps_for_mean.append(hist['steps_for_mean'].tolist())
        self._save_all_history(all_pathname)

    def _init_all_history(self):
        self.all_steps_in_episode = []
        self.all_rwds = []
        self.all_acts = []
        self.all_steps_for_mean = []
        self.all_Qss = []
        self.all_eval_rwds_in_episode = []
        self.all_eval_steps_in_episode = []
        self.all_eval_x = []

    def _save_all_history(self, all_pathname):
        np.savez(all_pathname + '.npz',
            all_steps_in_episode=self.all_steps_in_episode,
            all_rwds=self.all_rwds,
            all_acts=self.all_acts,
            all_Qss=self.all_Qss,
            all_eval_rwds=self.all_eval_rwds_in_episode,
            all_eval_steps=self.all_eval_steps_in_episode,
            all_eval_x=self.all_eval_x,
            all_steps_for_mean=self.all_steps_for_mean,
        )

    def _load_all_history(self, all_pathname):
        hist = np.load(all_pathname + '.npz')
        self.all_steps_in_episode = hist['all_steps_in_episode'].tolist()
        self.all_rwds = hist['all_rwds'].tolist()
        self.all_acts = hist['all_acts'].tolist()
        self.all_Qss = hist['all_Qss'].tolist()
        self.all_eval_rwds_in_episode = hist['all_eval_rwds'].tolist()
        self.all_eval_steps_in_episode = hist['all_eval_steps'].tolist()
        self.all_eval_x = hist['all_eval_x'].tolist()
        self.all_steps_for_mean = hist['all_steps_for_mean'].tolist()

    def show_history(self, window=10):
        steps = self.simhist.steps_for_mean
        plt.figure(figsize=(5,6))
        plt.subplots_adjust(hspace=0.6)

        plt.subplot(411)
        dat = self.simhist.mean_steps_in_episode
        plt.plot(steps, dat)
        plt.title('steps / episode (%d)' % dat[-1])
        plt.grid(True)

        plt.subplot(412)
        dat = self.simhist.mean_rwds
        plt.plot(steps, dat)
        plt.title('sum of rewards (%.2f)' % dat[-1])
        plt.grid(True)

        plt.subplot(413)
        dat = self.simhist.mean_acts
        plt.plot(steps, dat)
        plt.grid(True)
        plt.title('mean action (%.2f)' % dat[-1])

        plt.subplot(414)
        dat = np.array(self.hist_Qss)
        dat_n = dat.shape[0]
        plt.plot(steps, dat.reshape(dat_n, -1))
        plt.grid(True)
        plt.title('Qs')
        plt.show()

        return

    def _smooth(self, x, window=5):
        w = np.ones(window)/window
        x = np.convolve(x, w, mode='valid')
        return x

class SimHistory:
    def __init__(self):
        """
        エピソード毎にまとめる
        """
        self.reset()
    
    def reset(self, init_step = 0):
        self.mean_rwds = []
        self.rwds = []
        self.rwd  = 0

        self.mean_acts = []
        self.acts = []
        self.act = []

        self.mean_steps_in_episode = []
        self.steps_in_episode = []
        self.step_in_episode = 0

        self.steps_for_mean = []
        self.stepcnt = init_step

        self.times = []
        self.stime = time.time()
        self.start_time = 0
        self.time = 0

    def add(self, act, rwd, done):
        self.rwd += rwd
        self.act.append(act)
        self.step_in_episode += 1
        self.stepcnt += 1
        if done is True:
            self.rwds.append(self.rwd)
            self.rwd = 0
            self.acts.append(np.mean(self.act))
            self.act = []
            self.steps_in_episode.append(self.step_in_episode)
            self.step_in_episode = 0
    
    def record(self):
        self.mean_rwds.append(np.mean(self.rwds))
        self.rwds = []
        self.mean_acts.append(np.mean(self.acts))
        self.acts = []
        self.mean_steps_in_episode.append(np.mean(self.steps_in_episode))
        self.steps_in_episode = []
        self.steps_for_mean.append(self.stepcnt)
        self.times.append(self.start_time + time.time() - self.stime)

