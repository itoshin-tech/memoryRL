import numpy as np
import cv2
import random
import pdb


class Env():
    """
    複数のゴール（沼）と壁がある2D迷路問題
    """
    name = 'env_swanptour'

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
    

    def __init__(
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
            is_reset_after_subgoal=True,
            is_wall_observable=True,
            step_until_goal_hidden=-1,
        ):
        """
        パラメータ

        field_size:         フィールドの大きさ
        sight_size:         視野の大きさ(field_sizeよりも小さくする)
        max_time:           タイムリミット
        n_wall:             壁の数
        n_goal:             ゴールの数
        start_pos:          スタート地点 (n, n)
        start_dir:          スタート時の方向 (0, 1, 2, 3)
        reward_hit_wall:    壁に当たったときの報酬
        reward_move:        動きのコスト
        reward_goal:        ゴールに到達したときの報酬
        maze_type='random': 迷路タイプ 
            'random', 'Tmaze', 'Tmaze_large', 'Tmaze_one'
        second_visit_penalty:   一度到達すると通れなくなる(T, F)
        is_reset_after_subgoal: ゴールに行くとスタート地点にもどる(T, F)
        is_wall_observable:     壁が観察できる(T, F)
        step_until_goal_hidden: スタートからnステップ後ゴールが不可視(-1, n)
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
        self.is_reset_after_subgoal = is_reset_after_subgoal
        self.is_wall_observable = is_wall_observable
        self.step_until_goal_hidden = step_until_goal_hidden

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

    def reset(self):
        """  
        内部状態をリセットする
        """
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
            for mm in mline:
                if mm == 'w':
                    id = Env.ID_wall
                elif mm == ' ':
                    id = Env.ID_brank
                elif mm == 'g':
                    id = Env.ID_goal_new
                else:
                    raise ValueError()
                line.append(id)
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
                xx = np.random.randint(0, self.field_size)
                yy = np.random.randint(0, self.field_size)
                if not(xx == self.start_pos[0] and yy == self.start_pos[1]) \
                    and self._truefield[yy, xx] == Env.ID_brank:
                    self._truefield[yy, xx] = Env.ID_goal_new
                    break

        # wall
        for i in range(self.n_wall):
            for j in range(99):
                x = np.random.randint(0, self.field_size)
                y = np.random.randint(0, self.field_size)
                if not(self.agt_pos[0] == x and self.agt_pos[1] ==y) and \
                        self._truefield[y, x] == Env.ID_brank:
                    self._truefield[y, x] = Env.ID_wall
                    break
            if j >= 98:
                pdb.set_trace()
        return
    
    def _maze_check(self):
        """
        スタート地点から到達できるゴールの数を出力
        """
        fd = self._truefield
        fh, fw = fd.shape
        xa, ya = self.agt_pos
        dd = np.array([
                        [ 0, -1],
                        [-1,  0],
                        [ 1,  0],
                        [ 0,  1],
                        ]
            )

        ff = np.zeros((fh + 2, fw + 2), dtype=int)
        ff[1:-1, 1:-1] = fd
        Enable = 99
        ff[ya + 1, xa + 1] = Enable
        possible_goal = 0
        while True:
            is_change = False
            for iy in range(1, fh + 1):
                for ix in range(1, fw + 1):
                    if ff[iy, ix] == Enable:
                        for id in range(dd.shape[0]):
                            if ff[iy + dd[id, 0], ix + dd[id, 1]] == Env.ID_brank or \
                                ff[iy + dd[id, 0], ix + dd[id, 1]] == Env.ID_goal_new or \
                                ff[iy + dd[id, 0], ix + dd[id, 1]] == Env.ID_goal_visited:

                                if ff[iy + dd[id, 0], ix + dd[id, 1]] == Env.ID_goal_new:
                                    possible_goal += 1
                                    ff[iy + dd[id, 0], ix + dd[id, 1]] = Env.ID_goal_visited
                                    is_change = True
                                elif ff[iy + dd[id, 0], ix + dd[id, 1]] == Env.ID_goal_visited:
                                    if self.second_visit_penalty is False:
                                        ff[iy + dd[id, 0], ix + dd[id, 1]] = Enable
                                    else:
                                        pass
                                elif ff[iy + dd[id, 0], ix + dd[id, 1]] == Env.ID_brank:
                                    ff[iy + dd[id, 0], ix + dd[id, 1]] = Enable
                                    is_change = True
                                else:
                                    raise ValueError('err!')

            if is_change is False:
                break
        
        return possible_goal

    def step(self, action):
        """
        action にしたがって環境の状態を 1 step 進める
        """
        self.agt_state = 'move'  # render 用
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
                self._truefield[pos[1], pos[0]] = Env.ID_goal_visited
                reward = self.reward_goal
                self.n_visited_goal += 1
                if self.n_visited_goal == self.n_goal:
                    done = True
                    self.agt_pos = pos
                else:
                    done = False
                    if self.is_reset_after_subgoal is True:
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

        observation = self._make_observation()

        # render の用
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
        fs = self.field_size
        size = fs * 3
        # goal 観測用、まずフィールドの3倍の大きさのobs_goalを作る
        obs_goal = np.zeros((size, size), dtype=int)

        # agt_posを中心とした観測行列obs_goalを作成
        obs_goal[fs:fs * 2, fs:fs * 2] = around_goal
        ss = self.sight_size
        x = fs + self.agt_pos[0]
        y = fs + self.agt_pos[1]
        obs_goal = obs_goal[y-ss:y+ss+1, x-ss:x+ss+1]

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
        if self.is_wall_observable is True:
            obs_wall = np.ones((size, size), dtype=int)
            obs_wall[fs:fs * 2, fs:fs * 2] = around_wall
            obs_wall = obs_wall[y-ss:y+ss+1, x-ss:x+ss+1]
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
        unit = 50
        width = unit * self.field_size
        height = unit * self.field_size
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # 背景の描画
        cv2.rectangle(img, (0, 0), (width-1, height-1), col_brank, -1)

        # 壁とゴールの描画
        for ix in range(self.field_size):
            for iy in range(self.field_size):
                col = None
                if self._truefield[iy, ix] == Env.ID_wall:
                    # 壁
                    col = col_wall
                elif self._truefield[iy, ix] == Env.ID_goal_new:
                    # 未到達のゴール
                    col = col_goal_new
                    if self.step_until_goal_hidden > 0:
                        if self.time >= self.step_until_goal_hidden:
                            col = col_brank

                elif self._truefield[iy, ix] == Env.ID_goal_visited:
                    # 到達済のゴール
                    col = col_goal_visited
                    if self.step_until_goal_hidden > 0:
                        if self.time >= self.step_until_goal_hidden:
                            col = col_brank

                if col is not None:
                    r0 = (unit * ix, unit * iy)
                    r1 = (r0[0] + unit - 1,  r0[1] + unit - 1)
                    cv2.rectangle(img, r0, r1, col, -1)

        # ロボットの体の描画 -------------------
        # 状態で報酬で変える
        if self.agt_state == 'hit_wall':
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

        for iy in range(observation.shape[0]):
           for ix in range(observation.shape[1]):
                if observation[iy, ix] == 0:
                   col = (0, 0, 0)
                else:
                   col = (200, 200, 200)
                rate = 0.8  # 四角を小さくして隙間が見えるようにする
                r0 = (int(obs_unit * ix), int(obs_unit * iy))
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

    def seed(self, seed):
        np.random.seed(seed)

def show_obs(obs, act, rwd, done):
    print(obs)
    if act is None:
        pass
    else:
        print('act:%d, rwd:% .2f, done:%s' % (act, rwd, done))


if __name__ == '__main__':
    # env
    # 以下のパラメータを変えてテスト
    env = Env(
        field_size=5,   # フィールドの大きさ
        sight_size=2,   # 視野の大きさ(field_sizeよりも小さくする)
        max_time=30,    # タイムリミット
        n_wall=8,       # 壁の数
        n_goal=2,       # ゴールの数
        start_pos=(4, 4),   # スタート地点
        start_dir=0,        # スタート時の方向(0, 1, 2, 3)
        reward_hit_wall=-.2,    # 壁に当たったときの報酬
        reward_move=-.1,        # 動きのコスト
        reward_goal=1.0,        # ゴールに到達したときの報酬
        maze_type='Tmaze_one',     # 迷路タイプ 'random', 
        second_visit_penalty =False,
        is_reset_after_subgoal=True,
        is_wall_observable=True,
        step_until_goal_hidden=2,
    )
    obs=env.reset()

    print(obs)
    is_process = False
    done = False
    while True:
        img = env.render()
        cv2.imshow('img', img)
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
                print('start')
                obs = env.reset()
                act = None
                rwd = None
                done = False
            else:
                obs, rwd, done = env.step(act)

            show_obs(obs, act, rwd, done)

            is_process = False
    




    





   




    


