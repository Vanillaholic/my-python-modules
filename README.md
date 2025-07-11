# 自己编写的一些python常用模块
### frequency domian adaptive filter

frequency domain adaptive filter and a partial constraint edtion,could custom hop length and frame length 

- fdaf.py：频域自适应滤波
- pfdaf.py：partial constraint版本，可定义帧移和帧长

### noise
- noise.py：添加噪声

### wideband ambiguity
- ~~waf.py:宽带模糊函数~~(在老师提供版本的基础上改进，最好忽略)
- RDM.py:宽带模糊函数(本人编写的版本)

### sequence generator
- sequence.py:生成M序列和costas序列

### beamforming
- beamforming.py: 对信号进行波束形成
  - `CBF()`:常规波束形成
  - `CBF_gpu()`：使用cupy进行常规波束形成
