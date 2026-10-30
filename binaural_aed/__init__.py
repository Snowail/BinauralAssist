"""双声道 HRTF 音频增强：脚步方向/微声增强 + 枪声平滑。

模块
----
config    参数（阈值、增益、设备）
dsp       滤波器、包络、联动限幅、软削波
analysis  脚步/枪声瞬态检测
enhance   增强链与 BinauralPipeline
realtime  sounddevice 双工流 + 延迟报告
demo      合成音频自测与客观指标
"""

__version__ = "0.1.0"
