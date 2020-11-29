"""
sim_swanptour.py
"""
import sys
import numpy as np 
import cv2
import copy
import pdb

import env_swanptour as envnow
import trainer
import mngagthistory


if __name__ == '__main__':

    argvs = sys.argv
    if len(argvs) < 4:
        msg = '\n' + \
            '---- 使い方 ---------------------------------------\n' + \
            '3つのパラメータを指定して実行します\n\n' + \
            '> python sim_swanptour.py [agt_type] [maze_type] [process_type]\n' + \
            '\n' + \
            '[agt_type]: \tq, qnet, lstm, gru\n' + \
            '[maze_type]:\tg1w0_f5s4, g1w8_f5s2, g2w8_f5s2, ' + \
            'Tmaze_both, Tmaze_either\n' + \
            '[process_type]:\tlearn, more, graph, anime\n' + \
            '---------------------------------------------------'
        print(msg)
        sys.exit()
    
    agt_type = 'agt_' + argvs[1]
    maze_type = argvs[2]
    process_type = argvs[3]

    env_name = 'env_swanptour'

    # process_type /////////////////////

    if process_type == 'learn' or process_type == 'L':
        is_load_data = False
        is_learn = True
        is_show_all_graphs = True
        is_show_anime = False
        anime_n_episode = 0
    elif process_type == 'more' or process_type == 'M':
        is_load_data = True
        is_learn = True
        is_show_all_graphs = True
        is_show_anime = False
        anime_n_episode = 0
    elif process_type == 'graph' or process_type == 'G':
        is_load_data = False
        is_learn = False
        is_show_all_graphs = True
        is_show_anime = False
        anime_n_episode = 0
    elif process_type == 'anime' or process_type == 'A':
        is_load_data = True
        is_learn = False
        is_show_all_graphs = False
        is_show_anime = True
        anime_n_episode = 100
    else:
        print('process type が間違っています。')
        sys.exit()

    # maze_type paramter /////////////////////

    if maze_type == 'g1w0_f5s4':
        # qでの説明で使用
        env_prm = {
            'max_time': 15,
            'n_wall': 0,
            'n_goal': 1,
            'field_size': 5,
            'sight_size': 4,
            'start_pos': (2, 2),
            'reward_hit_wall': -0.2,
            'reward_move': -0.1,
            'reward_goal': 1,
            'maze_type': 'random',
            'is_wall_observable': False,
        }
        n_step = 5000
        show_Q_interval =200
        eary_stop_step = 4
        eary_stop_reward = 1.2
        agt_epsilon = 0.2
        agt_anime_epsilon = 0.0    

    elif maze_type == 'g1w8_f5s2':
        # qnetでの説明で使用
        env_prm = {
            'max_time': 15,
            'n_wall': 8,
            'n_goal': 1,
            'field_size': 5,
            'sight_size': 2,
            'start_pos': (2, 2),
            'reward_hit_wall': -0.2,
            'reward_move': -0.1,
            'reward_goal': 1,
            'maze_type': 'random',
        }
        n_step = 20000
        show_Q_interval =200
        eary_stop_step = 4.5
        # 5だと回り込みまではできない
        # 4.5はいける、回り込みできるときがある
        # 4には到達しない
        eary_stop_reward = None
        agt_epsilon = 0.4
        agt_anime_epsilon = 0.0    

    elif maze_type == 'g2w8_f5s2':
        # qnetでの説明で使用
        env_prm = {
            'max_time': 15,
            'n_wall': 8,
            'n_goal': 2,
            'field_size': 5,
            'sight_size': 2,
            'start_pos': (2, 2),
            'reward_hit_wall': -0.2,
            'reward_move': -0.1,
            'reward_goal': 1,
            'maze_type': 'random',
        }
        n_step = 20000
        show_Q_interval =200
        eary_stop_step = 4.5
        eary_stop_reward = None
        agt_epsilon = 0.4
        agt_anime_epsilon = 0.0    

    elif maze_type == 'Tmaze_one':
        # gruの説明で使用
        env_prm = {
            'max_time': 5,
            'n_goal': 1,
            'sight_size': 2,
            'reward_hit_wall': -0.2,
            'reward_move': -0.1,
            'reward_goal': 1,
            'maze_type': 'Tmaze_one',
            'step_until_goal_hidden': 2,
        }
        n_step = 20000
        show_Q_interval =200
        eary_stop_step = 4
        eary_stop_reward = None
        agt_epsilon = 0.4
        agt_anime_epsilon = 0.0

    elif maze_type == 'Tmaze':
        # gruの説明で使用
        env_prm = {
            'max_time': 12,
            'sight_size': 3,
            'reward_hit_wall': -0.2,
            'reward_move': -0.1,
            'reward_goal': 1,
            'maze_type': 'Tmaze',
        }
        n_step = 5000
        show_Q_interval =200
        eary_stop_step = 11
        eary_stop_reward = None
        agt_epsilon = 0.4
        agt_anime_epsilon = 0.0

    else:
        print('maze_type が間違っています。')
        sys.exit()

    # 学習用環境
    env = envnow.Env(**env_prm)
    obs = env.reset()
    obs2, _, _ = env.step(0)

    # 評価用環境
    eval_env_prm = copy.copy(env_prm)
    eval_env = envnow.Env(**eval_env_prm)


    # agent prameter  ///////////////////

    # agt common
    agt_prm = {
        'gamma': 0.9,
        'epsilon': agt_epsilon,
        'input_size': obs.shape,
        'n_action': env.n_action,
        'filepath': 'agt_data/sim_' + \
                    agt_type + '_' + \
                    env_name + '_' + \
                    maze_type
    }

    if agt_type == 'agt_q':
        agt_prm['init_val_Q'] = 0
        agt_prm['alpha'] = 0.1

    elif agt_type == 'agt_qnet':
        agt_prm['n_dense'] = 64
        agt_prm['n_dense2'] = None # ここを数値にすると

    elif agt_type == 'agt_lstm':
        agt_prm = copy.copy(agt_prm)
        agt_prm['n_dense'] = 64
        agt_prm['n_dense2'] = None
        agt_prm['n_lstm'] = 32
        agt_prm['memory_size'] = 100
        agt_prm['data_length_for_learn'] = 100
        agt_prm['learn_interval'] = 20
        agt_prm['epochs_for_train'] = 1

    elif agt_type == 'agt_gru':
        agt_prm = copy.copy(agt_prm)
        agt_prm['n_dense'] = 64
        agt_prm['n_dense2'] = None
        agt_prm['n_lstm'] = 32
        agt_prm['memory_size'] = 100
        agt_prm['data_length_for_learn'] = 100
        agt_prm['learn_interval'] = 20
        agt_prm['epochs_for_train'] = 1

    # simulation pramter /////////////////////
    sim_prm = {
        'n_step': n_step,
        'n_episode': -1,
        'show_Q_interval': show_Q_interval,
        'is_learn': is_learn,
        'is_animation': False,
        'show_delay': 0.5,
        'eval_n_step': -1,
        'eval_n_episode': 100,
        'eval_epsilon': 0.0,
        'eval_is_animation': False,
        'eval_show_delay': 0.0,
        'eary_stop_step': eary_stop_step,
        'eary_stop_reward': eary_stop_reward,
    }

    # animation pramter /////////////////////
    sim_anime_prm = {
        'n_step': -1,
        'n_episode': anime_n_episode,
        'is_learn': False, 
        'is_animation': True, 
        'show_delay': 0.2,
        'eval_n_step': -1,
        'eval_n_episode': -1,
    } 
  
    # trainer paramter ///////////////
    obss = [[obs.tolist(), obs2.tolist()]]
    trn_prm = {
        'obss': obss,
        'is_show_Q': False,
        'show_header': '',
    }

    # メイン ///////////////
    if (is_load_data is True) or \
        (is_learn is True) or \
        (sim_prm['is_animation'] is True):

        # エージェントインスタンス作成
        exec('from ' + agt_type + ' import Agt')
        agt = eval('Agt(**agt_prm)')
        agt.build_model()

        # trainer インスタンス作成
        trn = trainer.Trainer(agt, env, eval_env, **trn_prm)
        
        if is_load_data is True:
            # エージェントのデータロード
            try:
                agt.load_weights()
                trn.load_history(agt.filepath)
            except Exception as e:
                print(e)
                print('エージェントのパラメータがロードできません')
                sys.exit()
        
        # 学習
        if is_learn is True:
            trn.simulate(**sim_prm)
            agt.save_weights()
            trn.save_history(agt.filepath)

        # アニメーション
        if is_show_anime is True:
            agt.epsilon = agt_anime_epsilon
            trn.simulate(**sim_anime_prm)
    
    if is_show_all_graphs is True:
        # グラフ表示
        agt_names = [agt_type]
        filepaths = [agt_prm['filepath']]
        agth = mngagthistory.MngAgtHistory(agt_names, filepaths)
        agth.show_all_graphs(agt_names)
    
