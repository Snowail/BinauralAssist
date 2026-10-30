"""FPS 游戏多声道音频事件检测与方向估计原型。

模块
----
config      类别、7.1 声道布局、音频参数
capture     sounddevice 采集 / 录制
features    梅尔谱、SpecAugment、GCC-PHAT 空间特征
dataset     真实数据 Dataset + 合成数据
model       CNN-LSTM 多任务模型
train       训练循环
infer       离线/实时推理
realtime    延迟线 + 方向增益（dry-run 可验证）
"""

__version__ = "0.1.0"
