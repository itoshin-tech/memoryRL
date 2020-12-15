"""
env_swanptour.py
池巡りの環境
"""
import sys
import numpy as np
import cv2
import core


class Env(core.coreEnv):
    """
    複数のゴール（沼）と壁がある2D迷路問題
    """
    # 内部表現のID
    ID_brank = 0
    ID_agt = 1
    ID_wall = 2
    ID_goal_new = 3
    ID_goal_visited = 4

    # 方向
    dr = np.array([
            [0, -1],
            [-1, 0],
            [0, 1],
            [1, 0],
        ])

    # タスクリスト
    task_list = [
        'silent_ruin',
        'open_field',
        'many_swamp',
        'Tmaze_both',
        'Tmaze_either',
        'ruin_1swamp',
        'ruin_2swamp',
    ]


    def __init__(  # pylint:disable=too-many-arguments, too-many-locals
            self,
            field_size=5,
            sight_size=3,
            max_time=30,
            n_wall=1,
            n_goal=2,
            start_pos=(3, 3),
            start_dir=0,
            reward_hit_wall = -0.2,
            reward_move = -0.1,
            reward_goal = 1.0,
            maze_type='random',
            second_visit_penalty=False,
            reset_after_subgoal=True,
            erase_visited_goal=False,
            wall_observable=True,
            step_until_goal_hidden=-1,
        ):
        """
        Parameters
        ----------
        field_size :int
            フィールドの大きさ
        sight_size: int
            視野の大きさ(field_sizeよりも小さくする)
        max_time: int
            タイムリミット
        n_wall: int
            壁の数
        n_goal: int
            ゴールの数
        start_pos: (int, int)
            スタート地点
        start_dir: int (0, 1, 2, or 3)
            スタート時の方向
        reward_hit_wall: float
            壁に当たったときの報酬
        reward_move: float
            動きのコスト
        reward_goal: float
            ゴールに到達したときの報酬
        maze_type='random': str
            迷路タイプ
            'random', 'Tmaze', 'Tmaze_large', 'Tmaze_one'
        second_visit_penalty: bool
            一度到達すると通れなくなる
        reset_after_subgoal: bool
            ゴールに行くとスタート地点にもどる
        wall_observable: bool
            壁が観察できる
        step_until_goal_hidden: int
            -1: ずっと可視
            n>0: nステップ後に不可視
        """
        self.field_size = field_size
        self.sight_size = sight_size
        self.max_time = max_time
        self.n_wall = n_wall
        self.n_goal = n_goal
        self.start_pos = start_pos
        self.start_dir = start_dir
        self.reward_hit_wall = reward_hit_wall
        self.reward_move = reward_move
        self.reward_goal = reward_goal
        self.maze_type = maze_type
        self.second_visit_penalty = second_visit_penalty
        self.reset_after_subgoal = reset_after_subgoal
        self.erase_visited_goal = erase_visited_goal
        self.wall_observable = wall_observable
        self.step_until_goal_hidden = step_until_goal_hidden

        super().__init__()

        # 行動数
        self.n_action = 3

        # 変数
        self.agt_pos = None
        self.agt_dir = None
        self.n_visited_goal = 0  # ゴールを訪れた回数
        self.field = None  # フィールド行列
        self.done = None  # 最終状態だったらTrue
        self.reward = None
        self.action = None
        self.time = None
        self.is_first_step = True  # エピソードの最初のstepのみTrue
        self.agt_state = None  # render 用
        self._truefield = None

    def set_task_type(self, task_type):
        """
        task_type を指定して、parameterを一括設定する
        """
        if task_type == Env.task_list[0]:
            # silent_ruin
            self.field_size = 5
            self.sight_size = 2
            self.max_time = 25
            self.n_wall = None
            self.n_goal = None
            self.start_pos = None
            self.reward_hit_wall = -0.2
            self.reward_move = -0.1
            self.reward_goal = 1
            self.maze_type = 'fixed_maze01'
            self.second_visit_penalty = False
            self.reset_after_subgoal = False
            self.erase_visited_goal = True
            self.wall_observable = True
            self.step_until_goal_hidden = -1

        elif task_type == Env.task_list[1]:
            # open_field
            self.field_size = 5
            self.sight_size = 4
            self.max_time = 15
            self.n_wall = 0
            self.n_goal = 1
            self.start_pos = (2, 2)
            self.reward_hit_wall = -0.2
            self.reward_move = -0.1
            self.reward_goal = 1
            self.maze_type = 'random'
            self.second_visit_penalty = False
            self.reset_after_subgoal = True
            self.erase_visited_goal = False
            self.wall_observable = False
            self.step_until_goal_hidden = -1

        elif task_type == Env.task_list[2]:
            # many_swamp
            self.field_size = 7
            self.sight_size = 2
            self.max_time = 30
            self.n_wall = 4
            self.n_goal = 4
            self.start_pos = (3, 3)
            self.reward_hit_wall = -0.2
            self.reward_move = -0.1
            self.reward_goal = 1
            self.maze_type = 'random'
            self.second_visit_penalty = False
            self.reset_after_subgoal = False
            self.erase_visited_goal = True
            self.wall_observable = True
            self.step_until_goal_hidden = -1

        elif task_type == Env.task_list[3]:
            # Tmaze_both
            self.field_size = None
            self.sight_size = 3
            self.max_time = 20
            self.n_wall = None
            self.n_goal = None
            self.start_pos = None
            self.reward_hit_wall = -0.2
            self.reward_move = -0.1
            self.reward_goal = 1
            self.maze_type = 'Tmaze'
            self.second_visit_penalty = False
            self.reset_after_subgoal = True
            self.erase_visited_goal = False
            self.wall_observable = True
            self.step_until_goal_hidden = -1

        elif task_type == Env.task_list[4]:
            # Tmaze_either
            self.field_size = None
            self.sight_size = 2
            self.max_time = 5
            self.n_wall = None
            self.n_goal = None
            self.start_pos = None
            self.reward_hit_wall = -0.2
            self.reward_move = -0.1
            self.reward_goal = 1
            self.maze_type = 'Tmaze_one'
            self.second_visit_penalty = False
            self.reset_after_subgoal = True
            self.erase_visited_goal = False
            self.wall_observable = True
            self.step_until_goal_hidden = 2

        elif task_type == Env.task_list[5]:
            # ruin_1swamp
            self.field_size = 5
            self.sight_size = 2
            self.max_time = 10
            self.n_wall = 8
            self.n_goal = 1
            self.start_pos = (2, 2)
            self.reward_hit_wall = -0.2
            self.reward_move = -0.1
            self.reward_goal = 1
            self.maze_type = 'random'
            self.second_visit_penalty = False
            self.reset_after_subgoal = True
            self.erase_visited_goal = False
            self.wall_observable = True
            self.step_until_goal_hidden = -1

        elif task_type == Env.task_list[6]:
            # ruin_2swamp
            self.field_size = 5
            self.sight_size = 2
            self.max_time = 20
            self.n_wall = 8
            self.n_goal = 2
            self.start_pos = (2, 2)
            self.reward_hit_wall = -0.2
            self.reward_move = -0.1
            self.reward_goal = 1
            self.maze_type = 'random'
            self.second_visit_penalty = False
            self.reset_after_subgoal = True
            self.erase_visited_goal = False
            self.wall_observable = True
            self.step_until_goal_hidden = -1

        else:
            MSG = '[task_type] が違います\n'  + \
            '%s ではなく、\n' % task_type + \
            '以下のどれかを指定してください\n' + \
            '%s' % ', '.join(Env.task_list)
            print(MSG)
            sys.exit()


    def reset(self):
        self.done = False
        self.reward = None
        self.action = None
        self.is_first_step = True
        self.agt_state = 'move'  # render 用
        self.time = 0
        self.n_visited_goal = 0

        if self.maze_type == 'random':
            # 迷路をランダム生成
            for i in range(100):
                self._make_maze()
                # 解けないパターンが生成される場合があるのでチェックをする
                possible_goal = self._maze_check()
                if possible_goal == self.n_goal:
                    break
                if i >= 98:
                    raise ValueError('迷路が生成できません。壁の数を減らしてください。')

        elif self.maze_type == 'Tmaze':
            maze = [
                'wwwww',
                'g   g',
                'ww ww',
                'ww ww',
                'wwwww',
                ]
            self.n_goal = 2
            self.my_maze(maze, start_pos=(2, 3), start_dir=0)
        elif self.maze_type == 'Tmaze_large':
            maze = [
                'g   g',
                'ww ww',
                'ww ww',
                'ww ww',
                'ww ww',
                ]
            self.n_goal = 2
            self.my_maze(maze, start_pos=(2, 4), start_dir=0)
        elif self.maze_type == 'Tmaze_one':
            if np.random.rand() > .5:
                maze = [
                    'wwwww',
                    'wg  w',
                    'ww ww',
                    'ww ww',
                    'wwwww',
                    ]
            else:
                maze = [
                    'wwwww',
                    'w  gw',
                    'ww ww',
                    'ww ww',
                    'wwwww',
                    ]
            self.n_goal = 1
            self.my_maze(maze, start_pos=(2, 3), start_dir=0)
        elif self.maze_type == 'fixed_maze01':
            maze = [
                'wwg  ',
                ' w  w',
                ' g  w',
                'www  ',
                'w    ',
                ]
            self.n_goal = 2
            self.my_maze(maze, start_pos=(4, 4), start_dir=0)

        else:
            raise ValueError('maze_type が間違っています')


        observation = self._make_observation()
        return observation


    def my_maze(self, maze, start_pos=(0, 0), start_dir=0):
        """
        文字で表した迷路を行列に変換

        ' ': brank
        'g': goal
        'w': wall
        """
        myfield = []
        for mline in maze:
            line = []
            id_val = None
            for i in mline:
                if i == 'w':
                    id_val = Env.ID_wall
                elif i == ' ':
                    id_val = Env.ID_brank
                elif i == 'g':
                    id_val = Env.ID_goal_new
                else:
                    raise ValueError()
                line.append(id_val)
            myfield.append(line)
        self._truefield = np.array(myfield, dtype=int)
        self.field_size = self._truefield.shape[0]
        self.start_pos = start_pos
        self.start_dir = start_dir

        # start
        self.agt_pos = self.start_pos
        self.agt_dir = self.start_dir


    def _make_maze(self):
        """
        ランダムな迷路を生成
        """
        # start
        self.agt_pos = self.start_pos
        self.agt_dir = self.start_dir

        # field
        self._truefield = np.ones((self.field_size,) * 2, dtype=int) * Env.ID_brank

        # goal
        for _ in range(self.n_goal):
            while True:
                x_val = np.random.randint(0, self.field_size)
                y_val = np.random.randint(0, self.field_size)
                if not(x_val == self.start_pos[0] and y_val == self.start_pos[1]) \
                    and self._truefield[y_val, x_val] == Env.ID_brank:
                    self._truefield[y_val, x_val] = Env.ID_goal_new
                    break

        # wall
        for _ in range(self.n_wall):
            for j in range(99):
                x_val = np.random.randint(0, self.field_size)
                y_val = np.random.randint(0, self.field_size)
                if not(self.agt_pos[0] == x_val and self.agt_pos[1] ==y_val) and \
                        self._truefield[y_val, x_val] == Env.ID_brank:
                    self._truefield[y_val, x_val] = Env.ID_wall
                    break
            if j >= 98:
                print('壁の数が多すぎて迷路が作れません')
                sys.exit()

    def _maze_check(self):
        """
        スタート地点から到達できるゴールの数を出力
        """
        field = self._truefield
        f_h, f_w = field.shape
        x_agt, y_agt = self.agt_pos

        f_val = np.zeros((f_h + 2, f_w + 2), dtype=int)
        f_val[1:-1, 1:-1] = field
        enable = 99
        f_val[y_agt + 1, x_agt + 1] = enable
        possible_goal = 0
        while True:  # pylint:disable=too-many-nested-blocks
            is_change = False
            for i_y in range(1, f_h + 1):
                for i_x in range(1, f_w + 1):
                    if f_val[i_y, i_x] == enable:
                        f_val, is_change, reached_goal = \
                            self._count_update(f_val, i_x, i_y, enable)
                        possible_goal += reached_goal
            if is_change is False:
                break

        return possible_goal

    def _count_update(self, f_val, i_x, i_y, enable):
        d_agt = np.array([
                        [ 0, -1],
                        [-1,  0],
                        [ 1,  0],
                        [ 0,  1],
                        ]
            )
        is_change = False
        possible_goal = 0
        for i in range(d_agt.shape[0]):
            if f_val[i_y + d_agt[i, 0], i_x + d_agt[i, 1]] == Env.ID_brank or \
                f_val[i_y + d_agt[i, 0], i_x + d_agt[i, 1]] == Env.ID_goal_new or \
                f_val[i_y + d_agt[i, 0], i_x + d_agt[i, 1]] == Env.ID_goal_visited:

                if f_val[i_y + d_agt[i, 0], i_x + d_agt[i, 1]] == Env.ID_goal_new:
                    possible_goal += 1
                    f_val[i_y + d_agt[i, 0], i_x + d_agt[i, 1]] = Env.ID_goal_visited
                    is_change = True
                elif f_val[i_y + d_agt[i, 0], i_x + d_agt[i, 1]] == Env.ID_goal_visited:
                    if self.second_visit_penalty is False:
                        f_val[i_y + d_agt[i, 0], i_x + d_agt[i, 1]] = enable
                    else:
                        pass
                elif f_val[i_y + d_agt[i, 0], i_x + d_agt[i, 1]] == Env.ID_brank:
                    f_val[i_y + d_agt[i, 0], i_x + d_agt[i, 1]] = enable
                    is_change = True
                else:
                    raise ValueError('err!')

        return f_val, is_change, possible_goal


    def step(self, action):
        """
        action にしたがって環境の状態を 1 step 進める
        """
        self.agt_state = 'move'  # render 用
        done = None
        if action == 0:
            # 前進
            pos = self.agt_pos + Env.dr[self.agt_dir]
            if pos[0] < 0 or self.field_size <= pos[0] or \
                pos[1] < 0 or self.field_size <= pos[1]:
                # 範囲外に行こうとした
                self.agt_state = 'hit_wall'
                reward = self.reward_hit_wall
                done = False

            elif self._truefield[pos[1], pos[0]] == Env.ID_goal_new:
                # 訪れていないゴールに訪れた場合
                self.agt_state = 'goal_new'
                if self.erase_visited_goal is True:
                    self._truefield[pos[1], pos[0]] = Env.ID_brank
                else:
                    self._truefield[pos[1], pos[0]] = Env.ID_goal_visited
                reward = self.reward_goal
                self.n_visited_goal += 1
                if self.n_visited_goal == self.n_goal:
                    done = True
                    self.agt_pos = pos
                else:
                    done = False
                    if self.reset_after_subgoal is True:
                        self.agt_pos = self.start_pos
                        self.agt_dir = self.start_dir
                    else:
                        self.agt_pos = pos

            elif self._truefield[pos[1], pos[0]] == Env.ID_goal_visited:
                # 一度訪れたゴールに一度訪れた場合
                self.agt_state = 'goal_visited'
                if self.second_visit_penalty is True:
                    reward = self.reward_hit_wall
                else:
                    reward = self.reward_move
                    self.agt_pos = pos
                done = False

            elif self._truefield[pos[1], pos[0]] == Env.ID_brank:
                # 何もないので進める
                self.agt_state = 'brank'
                self.agt_pos = pos
                reward = self.reward_move
                done = False

            elif self._truefield[pos[1], pos[0]] == Env.ID_wall:
                # 壁に進もうとした
                self.agt_state = 'hit_wall'
                reward = self.reward_hit_wall
                done = False

            else:
                raise ValueError('Err!')

        elif action == 1:
            # 右に向く
            self.agt_dir = (self.agt_dir + 1) % 4
            reward = self.reward_move
            done = False

        elif action == 2:
            # 左に向く
            self.agt_dir = (self.agt_dir - 1) % 4
            reward = self.reward_move
            done = False

        else:
            raise ValueError('Err!')

        # 時間切れ
        self.time += 1
        if self.time >= self.max_time:
            reward = self.reward_hit_wall
            done = True
            self.agt_state = 'timeover'

        observation = self._make_observation()

        # render 用
        self.done = done
        self.reward = reward
        self.action = action

        return observation, reward, done

    def _make_observation(self):
        """
        現在の状態から、エージェントが受け取る入力情報を生成
        入力情報は自己を中心としたゴールと壁の位置
        """
        # make around
        around_wall = self._truefield.copy()
        around_wall[np.where(around_wall != Env.ID_wall)] = 0
        around_wall[around_wall > 0] = 1

        around_goal_new = self._truefield.copy()
        around_goal_new[around_goal_new != Env.ID_goal_new] = 0

        around_goal_visited = self._truefield.copy()
        around_goal_visited[around_goal_visited != Env.ID_goal_visited] = 0

        around_goal = around_goal_visited + around_goal_new
        around_goal[around_goal > 0] = 1

        if self.step_until_goal_hidden > 0:
            if self.time >= self.step_until_goal_hidden:
                around_goal[around_goal == 1] = 0

        # init field
        f_s = self.field_size
        size = f_s * 3
        # goal 観測用、まずフィールドの3倍の大きさのobs_goalを作る
        obs_goal = np.zeros((size, size), dtype=int)

        # agt_posを中心とした観測行列obs_goalを作成
        obs_goal[f_s:f_s * 2, f_s:f_s * 2] = around_goal
        s_s = self.sight_size
        x_val = f_s + self.agt_pos[0]
        y_val = f_s + self.agt_pos[1]
        obs_goal = obs_goal[y_val-s_s:y_val+s_s+1, x_val-s_s:x_val+s_s+1]

        # ロボットの方向に合わせて観測行列を回転
        if self.agt_dir == 3:
            obs_goal = np.rot90(obs_goal)
        elif self.agt_dir == 2:
            for _ in range(2):
                obs_goal = np.rot90(obs_goal)
        elif self.agt_dir == 1:
            for _ in range(3):
                obs_goal = np.rot90(obs_goal)

        # 同様に壁の観測行列を作成
        if self.wall_observable is True:
            obs_wall = np.ones((size, size), dtype=int)
            obs_wall[f_s:f_s * 2, f_s:f_s * 2] = around_wall
            obs_wall = obs_wall[y_val-s_s:y_val+s_s+1, x_val-s_s:x_val+s_s+1]
            if self.agt_dir == 3:
                obs_wall = np.rot90(obs_wall)
            elif self.agt_dir == 2:
                for _ in range(2):
                    obs_wall = np.rot90(obs_wall)
            elif self.agt_dir == 1:
                for _ in range(3):
                    obs_wall = np.rot90(obs_wall)

            obs = np.c_[obs_goal, obs_wall]
        else:
            obs = obs_goal

        return obs

    def render(self):
        """
        画面表示用の画像を生成
        ※エージェントの入力情報ではなくユーザー用
        """
        # フィールドの描画 --------------------
        # 色
        col_brank = (0, 255, 0)
        col_wall = (0, 0, 70)
        col_agt = (255, 255, 255)
        col_agt_miss = (0, 0, 255)
        col_agt_rwd = (50, 200, 50)
        col_agt_edge = (0, 0, 0)
        col_goal_new = (255, 100, 0)
        col_goal_visited = (255, 200, 100)
        col_agt_obs = (50, 50, 255)

        # 画像サイズ
        unit = int(250.0 / self.field_size)
        width = unit * self.field_size
        height = unit * self.field_size
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # 背景の描画
        cv2.rectangle(img, (0, 0), (width-1, height-1), col_brank, -1)

        # 壁とゴールの描画
        for i_x in range(self.field_size):
            for i_y in range(self.field_size):
                col = None
                if self._truefield[i_y, i_x] == Env.ID_wall:
                    # 壁
                    col = col_wall
                elif self._truefield[i_y, i_x] == Env.ID_goal_new:
                    # 未到達のゴール
                    col = col_goal_new
                    if self.step_until_goal_hidden > 0:
                        if self.time >= self.step_until_goal_hidden:
                            col = col_brank

                elif self._truefield[i_y, i_x] == Env.ID_goal_visited:
                    # 到達済のゴール
                    col = col_goal_visited
                    if self.step_until_goal_hidden > 0:
                        if self.time >= self.step_until_goal_hidden:
                            col = col_brank

                if col is not None:
                    r0 = (unit * i_x, unit * i_y)
                    r1 = (r0[0] + unit - 1,  r0[1] + unit - 1)
                    cv2.rectangle(img, r0, r1, col, -1)

        # ロボットの体の描画 -------------------
        # 状態で報酬で変える
        if self.agt_state == 'hit_wall' or \
            self.agt_state == 'timeover':
            col = col_agt_miss
        elif self.agt_state == 'goal_new':
            col = col_agt_rwd
        elif self.agt_state == 'goal_visited':
            if self.second_visit_penalty is True:
                col = col_agt_miss
            else:
                col = col_agt
        else:
            col = col_agt
        radius = int(unit * 0.35)
        r0 = (self.agt_pos + np.array([0.5, 0.5]))* unit
        r0 = r0.astype(int)
        cv2.circle(img, tuple(r0), radius, col, -1)
        cv2.circle(img, tuple(r0), radius, col_agt_edge, 2)

        # ロボットの方向の描画
        radius = int(unit * 0.2)
        i = self.agt_dir
        r1 = np.array(r0) + unit * 0.25 * Env.dr[i, :]
        r1 = r1.astype(int)
        cv2.circle(img, tuple(r1), radius, col_agt_edge, -1)

        # 観測値の描画 ----------------------
        observation = self._make_observation()
        obs_ih, obs_iw = observation.shape
        obs_unit = height / obs_ih  # 画像の縦の長さがフィールドと同じになるように obs_unitを決める
        obs_width = int(obs_unit * obs_iw)
        img_obs = np.zeros((height, obs_width, 3), dtype=np.uint8)

        for i_y in range(observation.shape[0]):
            for i_x in range(observation.shape[1]):
                if observation[i_y, i_x] == 0:
                    col = (0, 0, 0)
                else:
                    col = (200, 200, 200)
                rate = 0.8  # 四角を小さくして隙間が見えるようにする
                r0 = (int(obs_unit * i_x), int(obs_unit * i_y))
                r1 = (int(r0[0] + obs_unit * rate),  int(r0[1] + obs_unit * rate))
                cv2.rectangle(img_obs, r0, r1, col, -1)

        # 中心にロボットを描画
        cy = int((obs_ih - 1) / 2)
        if obs_iw == obs_ih * 2:
            cxs = (cy, obs_ih + cy)
        else:
            cxs = (cy, )

        col = col_agt_obs  # ロボットの色
        for cx in cxs:
            radius = int(obs_unit * 0.35)
            r0 = (int(obs_unit * (cx + rate * 0.5)),
                  int(obs_unit * (cy + rate * 0.5)))
            cv2.circle(img_obs, r0, radius, col, 2)

            # エージェントの方向の描画
            radius = int(obs_unit * 0.2)
            r1 = np.array(r0) + obs_unit * 0.25 * Env.dr[0, :]
            r1 = r1.astype(int)
            cv2.circle(img_obs, tuple(r1), radius, col, -1)

        mgn_w = 10  # フィールドと観測値の境界線の太さ
        mgn_col = (200, 200, 0)
        img_mgn = np.zeros((height, mgn_w, 3), dtype=np.uint8)
        cv2.rectangle(
            img_mgn,
            (0, 0), (mgn_w, height), mgn_col, -1)

        img_out = cv2.hconcat([img, img_mgn, img_obs])

        return img_out


def show_obs(observation, action, reward, dones):
    """
    変数を表示
    """
    if act is not None:
        print(observation)
        print('act:%d, rwd:% .2f, done:%s' % (action, reward, dones))
    else:
        print('start')
        print(observation)


if __name__ == '__main__':

    argvs = sys.argv

    if len(argvs) < 2:
        MSG = '\n' + \
            '---- 操作方法 -------------------------------------\n' + \
            '[task type] を指定して実行します\n' + \
            '> python env_swanptour.py [task_type]\n' + \
            '[task_type]\n' + \
            '%s\n' % ', '.join(Env.task_list)
        print(MSG)
        sys.exit()

    env = Env()
    env.set_task_type(argvs[1])
    MSG =  '---- 操作方法 -------------------------------------\n' + \
           '[e] 前に進む [s] 左に90度回る [f] 右に90度回る\n' + \
           '[q] 終了\n' + \
           '全ての池に訪れるとクリア、次のエピソードが開始\n' + \
           '---------------------------------------------------'
    print(MSG)
    print('[task_type]: %s\n' % argvs[1])
    is_process = False
    obs = env.reset()
    act = None
    rwd = None
    done = False
    show_obs(obs, act, rwd, done)
    while True:
        image = env.render()
        cv2.imshow('env', image)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        if key == ord('e'):
            act = 0
            is_process = True

        if key == ord('f'):
            act = 2
            is_process = True

        if key == ord('s'):
            act = 1
            is_process = True

        if is_process is True:
            if done is True:
                obs = env.reset()
                act = None
                rwd = None
                done = False
            else:
                obs, rwd, done = env.step(act)

            show_obs(obs, act, rwd, done)

            is_process = False
