"""
sim_swanptour.py
池巡りの実行ファイル
"""
import sys
import copy

import env_swanptour as envnow
from env_swanptour import TaskType
import trainer
import mng_agt_history


if __name__ == '__main__':

    argvs = sys.argv
    if len(argvs) < 4:
        MSG = '\n' + \
            '---- 使い方 ---------------------------------------\n' + \
            '3つのパラメータを指定して実行します\n\n' + \
            '> python sim_swanptour.py [agt_type] [task_type] [process_type]\n\n' + \
            '[agt_type]\t: q, qnet, lstm, gru\n' + \
            '[task_type]\t: %s\n' % ', '.join([t.name for t in TaskType]) + \
            '[process_type]\t:learn/L, more/M, graph/G, anime/A\n' + \
            '例 > python sim_swanptour.py q open_field L\n' + \
            '---------------------------------------------------'
        print(MSG)
        sys.exit()

    agt_type = argvs[1]

    task_type = TaskType.Enum_of(argvs[2])
    if task_type is None:
        MSG = '\n' + \
            '[task type] が異なります。以下から選んで指定してください。\n' + \
            '%s\n' % ', '.join([t.name for t in TaskType])
        print(MSG)
        sys.exit()

    process_type = argvs[3]

    ENV_NAME = 'env_swanptour'

    # process_type /////////////////////

    if process_type in ('learn', 'L'):
        IS_LOAD_DATA = False
        IS_LEARN = True
        IS_SHOW_ALL_GRAPHS = True
        IS_SHOW_ANIME = False
        ANIME_N_EPISODE = 0
    elif process_type in ('more', 'M'):
        IS_LOAD_DATA = True
        IS_LEARN = True
        IS_SHOW_ALL_GRAPHS = True
        IS_SHOW_ANIME = False
        ANIME_N_EPISODE = 0
    elif process_type in ('graph', 'G'):
        IS_LOAD_DATA = False
        IS_LEARN = False
        IS_SHOW_ALL_GRAPHS = True
        IS_SHOW_ANIME = False
        ANIME_N_EPISODE = 0
        print('グラフ表示を終了するには[q]を押します。')
    elif process_type in ('anime', 'A'):
        IS_LOAD_DATA = True
        IS_LEARN = False
        IS_SHOW_ALL_GRAPHS = False
        IS_SHOW_ANIME = True
        ANIME_N_EPISODE = 100
        print('アニメーションを途中で止めるには[q]を押します。')
    else:
        print('process type が間違っています。')
        sys.exit()

    # task_type paramter /////////////////////
    if task_type == TaskType.silent_ruin:
        N_STEP = 5000
        SHOW_Q_INTERVAL =200
        EARY_STOP_STEP = 15
        EARY_STOP_REWARD = None
        AGT_EPSILON = 0.4
        AGT_ANIME_EPSILON = 0.0

    elif task_type == TaskType.open_field:
        N_STEP = 5000
        SHOW_Q_INTERVAL =200
        EARY_STOP_STEP = 4
        EARY_STOP_REWARD = 1.2
        AGT_EPSILON = 0.2
        AGT_ANIME_EPSILON = 0.0

    elif task_type == TaskType.many_swamp:
        N_STEP = 5000
        SHOW_Q_INTERVAL =1000
        EARY_STOP_STEP = 22
        EARY_STOP_REWARD = 1.4
        AGT_EPSILON = 0.4
        AGT_ANIME_EPSILON = 0.0

    elif task_type == TaskType.Tmaze_both:
        N_STEP = 5000
        SHOW_Q_INTERVAL = 200
        EARY_STOP_STEP = 11
        EARY_STOP_REWARD = None
        AGT_EPSILON = 0.4
        AGT_ANIME_EPSILON = 0.0

    elif task_type == TaskType.Tmaze_either:
        N_STEP = 5000
        SHOW_Q_INTERVAL = 200
        EARY_STOP_STEP = 4
        EARY_STOP_REWARD = None
        AGT_EPSILON = 0.4
        AGT_ANIME_EPSILON = 0.0

    elif task_type == TaskType.ruin_1swamp:
        N_STEP = 5000
        SHOW_Q_INTERVAL =1000
        EARY_STOP_STEP = 4.5
        EARY_STOP_REWARD = None
        AGT_EPSILON = 0.4
        AGT_ANIME_EPSILON = 0.0

    elif task_type == TaskType.ruin_2swamp:
        N_STEP = 5000
        SHOW_Q_INTERVAL =1000
        EARY_STOP_STEP = 4.5
        EARY_STOP_REWARD = None
        AGT_EPSILON = 0.4
        AGT_ANIME_EPSILON = 0.0

    else:
        N_STEP = 5000
        SHOW_Q_INTERVAL =1000
        EARY_STOP_STEP = None
        EARY_STOP_REWARD = None
        AGT_EPSILON = 0.4
        AGT_ANIME_EPSILON = 0.0
        print('シミュレーションににデフォルトパラメータを設定しました。')

    # 学習用環境
    env = envnow.Env()
    env.set_task_type(task_type)
    obs = env.reset()
    obs2, _, _ = env.step(0)

    # 評価用環境
    eval_env = envnow.Env()
    eval_env.set_task_type(task_type)

    # agent prameter  ///////////////////
    # agt common
    agt_prm = {
        'gamma': 0.9,
        'epsilon': AGT_EPSILON,
        'input_size': obs.shape,
        'n_action': env.n_action,
        'filepath': 'agt_data/sim_' + \
                    ENV_NAME + '_' + \
                    agt_type + '_' + \
                    task_type.name
    }

    if agt_type == 'q':
        agt_prm['init_val_Q'] = 0
        agt_prm['alpha'] = 0.1

    elif agt_type == 'qnet':
        agt_prm['n_dense'] = 64
        agt_prm['n_dense2'] = None  # 数値にすると1層追加

    elif agt_type == 'lstm':
        agt_prm = copy.copy(agt_prm)
        agt_prm['n_dense'] = 64
        agt_prm['n_dense2'] = None  # 数値にすると1層追加
        agt_prm['n_lstm'] = 32
        agt_prm['memory_size'] = 100
        agt_prm['data_length_for_learn'] = 100
        agt_prm['learn_interval'] = 20
        agt_prm['epochs_for_train'] = 1

    elif agt_type == 'gru':
        agt_prm = copy.copy(agt_prm)
        agt_prm['n_dense'] = 64
        agt_prm['n_dense2'] = None  # 数値にすると1層追加
        agt_prm['n_lstm'] = 32
        agt_prm['memory_size'] = 100
        agt_prm['data_length_for_learn'] = 100
        agt_prm['learn_interval'] = 20
        agt_prm['epochs_for_train'] = 1

    # simulation pramter /////////////////////
    sim_prm = {
        'N_STEP': N_STEP,
        'n_episode': -1,
        'SHOW_Q_INTERVAL': SHOW_Q_INTERVAL,
        'IS_LEARN': IS_LEARN,
        'is_animation': False,
        'show_delay': 0.5,
        'eval_N_STEP': -1,
        'eval_n_episode': 100,
        'eval_epsilon': 0.0,
        'eval_is_animation': False,
        'eval_show_delay': 0.0,
        'EARY_STOP_STEP': EARY_STOP_STEP,
        'EARY_STOP_REWARD': EARY_STOP_REWARD,
    }

    # animation pramter /////////////////////
    sim_anime_prm = {
        'N_STEP': -1,
        'n_episode': ANIME_N_EPISODE,
        'IS_LEARN': False,
        'is_animation': True,
        'show_delay': 0.2,
        'eval_N_STEP': -1,
        'eval_n_episode': -1,
    }

    # trainer paramter ///////////////
    obss = [[obs.tolist(), obs2.tolist()]]
    trn_prm = {
        'obss': obss,
        'is_show_Q': False,
        'show_header': '%s %s ' % (agt_type, task_type.name),
    }

    # メイン ///////////////
    if (IS_LOAD_DATA is True) or \
        (IS_LEARN is True) or \
        (sim_prm['is_animation'] is True):

        # エージェントをインポートしてインスタンス作成
        if agt_type == 'q':
            from agt_q import Agt  # pylint:disable=unused-import
        elif agt_type == 'qnet':
            from agt_qnet import Agt
        elif agt_type == 'lstm':
            from agt_lstm import Agt
        elif agt_type == 'gru':
            from agt_gru import Agt
        else:
            print('agt_type が間違っています')
            sys.exit()

        agt = Agt(**agt_prm)
        agt.build_model()

        # trainer インスタンス作成
        trn = trainer.Trainer(agt, env, eval_env, **trn_prm)

        if IS_LOAD_DATA is True:
            # エージェントのデータロード
            try:
                agt.load_weights()
                trn.load_history(agt.filepath)
            except:  # pylint: disable=bare-except
                print('エージェントのパラメータがロードできません')
                sys.exit()

        # 学習
        if IS_LEARN is True:
            trn.simulate(**sim_prm)
            agt.save_weights()
            trn.save_history(agt.filepath)

        # アニメーション
        if IS_SHOW_ANIME is True:
            agt.epsilon = AGT_ANIME_EPSILON
            trn.simulate(**sim_anime_prm)

    if IS_SHOW_ALL_GRAPHS is True:
        # グラフ表示
        agt_names = [agt_type]
        filepaths = [agt_prm['filepath']]
        agth = mng_agt_history.MngAgtHistory(agt_names, filepaths)
        agth.show_all_graphs(agt_names)
