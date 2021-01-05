# memoryRL とは

このmemoryRLでは、例えば、以下のような強化学習タスクを学習させることができます。GPUのない普通のノートPCで学習させることが可能です。筆者の古いノートPCでも4分弱しかかかりませんでした (qnet, many_swamp, 33000 qstep, 3分40秒)。

※筆者のノートPCのCPUは、Intel Core i7-6600UでGPUなし。i7でも第6世代なので結構遅く、i5-7500Uの方が上のようです。 [参考：CPU性能比較表 | 最新から定番のCPUまで簡単に比較](https://pcrecommend.com/cpu/)

![](image/20201206_many_swamp01.gif)

タスクは、ロボットを動かして全てのゴール（水色の4つの領域）に訪れるとクリアというものです。マップはエピソード毎にランダムに生成されます。

ロボットは2マスまでの周囲しか見えない設定なのですが、「壁はよける」、「ゴールに近づいたらそちらに曲がる」、「何もなければまっすぐ進む」、のようなアルゴリズムで動いているように見えます。このアルゴリズムを強化学習が作ったのです。マップがランダムに変わっても、追加学習をすることなく対応しています。

このmemoryRLプロジェクトでは、このタスクを含む7つのバリエーションのタスクを、4種類の強化学習エージェントで試すことができます。

学習の原理、実装方法など、更に詳しい解説は、以下を参考にしてください。

http://itoshi.main.jp/tech/0100-rl/rl_introduction/


# 環境構築

ここでは、windows 10 を想定し、pythonでmemoryRLを動かすための環境構築を説明します。

## Anaconda インストール
まず、pythonの基本的な環境として、anaconda をインストールします。

参考
<a href="https://www.python.jp/install/anaconda/windows/install.html">Windows版Anacondaのインストール</a>

## 仮想環境作成
次に、特定のpythonのバージョンやモジュールのバージョンの環境を設定するために、conda で仮想環境を作成します。

参考
<a href="https://qiita.com/ozaki_physics/items/985188feb92570e5b82d">【初心者向け】Anacondaで仮想環境を作ってみる</a>
<a href="https://www.python.jp/install/anaconda/conda.html">python japan, Conda コマンド</a>

スタートメニューからAnaconda Powershell Prompt を立ち上げます。

以下のコマンドで、mRLという名前の仮想環境をpython3.6 で作成します。
```
(base)> conda create -n mRL python=3.6
```

以下で、mRLをアクティベートします（仮想環境に入る）。
```
(base)> conda activate mRL
```


tensorflow, numpy, h5py をバージョン指定でインストールします。
```
(mRL)> pip install tensorflow==1.12.0 numpy==1.16.1 h5py==2.10.0
```

opencv-pythonとmatplotlib は、最新のバージョンでインストールします。
```
(mRL)> pip install opencv-python matplotlib
```

今後、この仮想環境 mRL に入ることで(> conda activate mRL)、今インストールしたモジュール（ライブラリ）の環境で、プログラムを動かすことができます。

仮想環境から抜けるには以下のようにします。
```
(mRL)> conda deactivate
```

# memoryRLのダウンロードと展開

gitが使える方は、以下のコマンドでクローンすれば完了です。
```
(mRL)> git clone https://github.com/itoshin-tech/memoryRL.git
```

以下、git を使っていない人用の説明です。

ブラウザで、次ののURLに行きます（このHPです）。
https://github.com/itoshin-tech/memoryRL

Code から、Download zip を選び、PCの適当な場所(C:\myWorks\ を想定)に保存して解凍すると、memoryRL-master というフォルダが作られます

![](image/github.jpg)


これで準備ＯＫです。

# memoryRLの実行

memoryRLを展開したフォルダーに入ります。
```
(mRL)> cd C:\[解凍したディレクトリ]\memoryRL-master\  
```

sim_swanptour.py を以下のコマンドで実行します。
```
> python sim_swanptour.py
```

すると以下のように使い方が表示されます。
```
---- 使い方 ---------------------------------------
3つのパラメータを指定して実行します

> python sim_swanptour.py [agt_type] [task_type] [process_type]

[agt_type]      : q, qnet, lstm, gru
[task_type]     :silent_ruin, open_field, many_swamp, 
Tmaze_both, Tmaze_either, ruin_1swamp, ruin_2swamp, 
[process_type]  :learn/L, more/M, graph/G, anime/A
例 > python sim_swanptour.py q open_field L
---------------------------------------------------
```

説明にあるように、python sim_swanptour.py の後に3つのパラメータをセットして使います。

最後に図解しますので、ここでは簡単に説明します。

+ [agt_type]　強化学習のアルゴリズムを指定します。
  + q: Q学習
  + qnet: ニューラルネットを使ったQ学習
  + lstm: LSTMを使った短期記憶付きQ学習
  + gru: GRUを使った短期記憶付きQ学習
+ [task_type]　タスクのタイプ。全てのタスクにおいて全ての青いゴールに辿り着けばクリア
  + silent_ruin: マップ固定、ゴール数2
  + open_field: 壁なし、ゴール数1。ゴールの位置はランダムに変わる。
  + many_swamp: 壁あり、ゴール数4。配置がランダムに変わる。
  + Tmaze_both: T迷路。短期記憶が必要。
  + Tmaze_either: T迷路。短期記憶が必要。
  + ruin_1swamp: 壁あり、ゴール数1。
  + ruin_2swamp: 壁あり、ゴール数2。高難易度。
+ [process type] プロセスの種類
  + learn/L: 初めから学習する
  + more/M: 追加学習をする
  + graph/G: 学習曲線を表示する
  + anime/A: タスクを解いている様子をアニメーションで表示

以下、<strong class="marker-yellow">qnet</strong> (ニューラルネットを使ったQ学習) に <strong class="marker-yellow">many_swamp</strong> のタスクを学習させる場合を説明します。

学習を開始するので、最後のパラメータは、<strong class="marker-yellow">more か L</strong>にします。

```
(mRL)> python sim_swanptour.py qnet many_swamp L
```

すると、以下のようにコンソールに学習過程の評価が表示され、全5000 stepの学習が行われます。
```
qnet many_swamp  1000 --- 5 sec, eval_rwd -3.19, eval_steps  30.00
qnet many_swamp  2000 --- 9 sec, eval_rwd -0.67, eval_steps  28.17
qnet many_swamp  3000 --- 14 sec, eval_rwd -0.21, eval_steps  26.59
qnet many_swamp  4000 --- 18 sec, eval_rwd -1.27, eval_steps  28.72
qnet many_swamp  5000 --- 23 sec, eval_rwd -1.29, eval_steps  28.90
```

1000回に1回、評価のプロセスがあり、そこで、eval_rwdとeval_stepが計算されます。eval_rwd は、その時の1エピソード中の平均報酬、eval_steps は平均step数です。評価は、行動選択のノイズは0にして行われます。

eval_rwdやeval_step を指標として学習が目標値に進んだときにも学習は終了するようになっています（EARY_STOP）。

最後に以下のような学習過程のグラフ（eval_rwd, eval_steps）が表示されます。
[q]を押すとグラフが消え終了します。

![](image/qnet_5000.png)


学習の結果後の<strong class="marker-yellow">動作アニメーションを見る</strong>には、最後のパラメータを<strong class="marker-yellow">anime か A</strong>にします。

```
(mRL)> python sim_swanptour.py qnet many_swamp A
```
すると、以下のようなアニメーションが表示されます。
![](image/20201212_qnet_many_5000.gif)

100エピソードが終わると終了します。[q]を押すと途中終了します。


アニメーションを見ると、適切に動けていなことが分かります。まだ学習が足りないのです（アニメーションの中央から右側の白黒の図はエージェントへの入力を表しています）。

そこで、<strong class="marker-yellow">追加学習</strong>します。最後のパラメータを<strong class="marker-yellow">more か M</strong>にして実行します（初めから学習する場合は L を使います）。

```
(mRL)> python sim_swanptour.py qnet many_swamp M
```
　
このコマンドを数回繰り返します。EARY_STOPが表示されて途中終了すると、学習が良いところまで進んだといえます。many_swampは、eval_rwd が1.4以上または、eval_steps が22以下になるとEARY_STOPとなります。

```
qnet many_swamp  1000 --- 5 sec, eval_rwd  0.55, eval_steps  24.25
qnet many_swamp  2000 --- 9 sec, eval_rwd  0.92, eval_steps  23.52
qnet many_swamp  3000 --- 14 sec, eval_rwd  1.76, eval_steps  21.69
EARY_STOP_STEP 22 >= 21
```

グラフが最後に表示されます。エピソード当たりの報酬(rwd)が増加し、ステップ数(Steps)が減少していることから、学習が進んでいたことが分かります。

![](image/qnet_30000.png)

アニメーションを見てみましょう。

```
(mRL)> python sim_swanptour.py qnet many_swamp A
```

たまに失敗しますが、だいたいうまくいっているようです。

![](image/20201212_qnet_many_30000.gif)

今までに学習させた<strong class="marker-yellow">グラフを表示</strong>するには、最後のパラメータを<strong class="marker-yellow">graph か G</strong>にします。

```
(mRL)> python sim_swanptour.py qnet many_swamp G
```


以上がsim_swamptour.py（池巡り）の使い方の説明です。

# 強化学習アルゴリズムの種類

[agt_type] で指定できる強化学習アルゴリズム（エージェント）は、 q, qnet, lstm, gru の4種類です。ここではその特徴を簡単に説明します。

## q：通常のQ学習 
基本の<strong class="marker-yellow">Q学習アルゴリズム</strong>です。各観察に対して各行動のQ値を変数（Qテーブル）に保存し、更新していきます。観測値は500個まで登録できる設定にしています。それ以上のパターンが観測されたらメモリーオーバーで強制終了となります。

![](image/agt_q.png)

## qnet：ニューラルネットを使ったQ学習アルゴリズム
<strong class="marker-yellow">ニューラルネットワーク</strong>でQ値を出力するよう学習します。入力は観測値で、出力は3つ値です。この3つの値が各行動のQ値に対応します。中間層は64個のReLUユニットです。
未知の観測値に対してもQ値を出力することができます。

![](image/agt_qnet.png)

## lstm/gru: 記憶ユニットを使ったQ学習アルゴリズム

過去の入力にも依存した反応が可能な<strong class="marker-yellow">記憶ユニット（LSTM または GRU）</strong>を加えたモデルです。LSTMを使ったアルゴリズムをlstm、GRUを使ったアルゴリズムをgruとしています。

qやqnetは、観測値が同じであれば同じ行動しか出力することしかできませんが、lstmやgruはモデルは、<strong class="marker-yellow">過去の入力が異なれば今の観測値が同じでも異なる行動を出力することが原理的に可能</strong>です。

LSTMは自然言語処理のモデルでもよく使用されている記憶ユニットです。GRUはLSTMをシンプルにしたモデルです。

![](image/agt_lstm.png)

LeRU を64個、LSTMまたはGRU を32個を使用しています。


# タスクの種類
[task_type] で指定できるタスクは、silent_ruin, open_field, many_swamp, Tmaze_both, Tmaze_either, ruin_1swamp, ruin2swamp の7種類です。ここではその特徴を簡単に説明します。

## 全タスクで共通のルール

全てのタスクで共通しているのは、ロボットが<strong class="marker-yellow">全てのゴール（青のマス）に訪れたらクリア</strong>、というルールです。

アルゴリズムが受け取る情報は、<strong class="marker-yellow">ロボットを中心とした、限られた視野におけるゴールと壁の情報</strong>です。各タスクの図の、右側の白黒の図がその情報に対応します。

報酬は、初めてのゴールにたどり着くと+1.0、壁に当たると-0.2、それ以外のステップで-0.1となります。

## silent_ruin

ゴールは2か所ありますが、マップは常に同じです。そのために、観測のバリエーションは限られており、q でも学習が可能です。

![](image/silent_ruin.png)


## open_field

壁はありませんが、ゴールの場所はエピソード毎にランダムに変わります。しかし、壁がないので観測のバリエーションは限られており、qでも学習が可能です。

![](image/open_field.png)

## many_swamp

ゴールと壁の位置がエピソード毎にランダムに決まるために、観測のバリエーションが多く、qではメモリーオーバーとなってしまい学習ができません。qnet での学習が可能です。

![](image/many_swamp.png)

## Tmaze_both

普通の強化学習ではできない問題です。

マップは固定ですが、片方のゴールにたどり着くと、スタート地点に戻されます。そして、訪れたゴールも見えたままです。この状態で、次は別な方のゴールに進まなければなりません。

スタート地点にロボットがいるとき、スタート直後でも、片方のゴールに訪れた後でも同じ観測となります。そのために、同じ観測に対して同じ行動しか選べないqやqnet は、適切な行動を学習種ることができません。過去の履歴で行動を変えることができる <strong class="marker-yellow">gru と lstm のみが学習可能</strong>です。

![](image/Tmaze_both.png)


## Tmaze_either

このTmazeは、左右のどちらかでゴールが出現しますが、2ステップ後にゴールが見えなくなります。

ロボットは2ステップでT迷路の分岐路に来ることになりますが、ゴールが見えていた方を覚えておき、そちらに向かうことが必要です。このタスクも、qやqnet ではできません。<strong class="marker-yellow">gruとlstmのみが学習可能</strong>です。

![](image/Tmaze_eigher.png)

## ruin_1swamp

壁が8個、ゴールは1つで、配置がランダムに変わります。回り込んでゴールにたどり着かなければならない場合もあり難易度は高めです。

![](image/ruin_1swamp.png)

## ruin_2swamp

最高難易度のタスクです。ゴールは2つで、片方にたどり着くとスタート地点に戻されます。ゴールは訪れても消えません。Tmaze_both のランダムバージョンのようなタスクです。gruやlstmでも満足にはできませんでした。

ゴールを更に増やしたりフィールドを大きくすることで、更に難易度を上げることができます。将来このようなタスクでもしっかり解けるような強化学習アルゴリズムを開発したいものです。

![](image/ruin_2swamp.png)

# 参考
学習の原理、実装方法など、更に詳しい解説は、以下を参考にしてください。

http://itoshi.main.jp/tech/0100-rl/rl_introduction/

