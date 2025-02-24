""" frequency domain adaptive filter """

import numpy as np
from numpy.fft import rfft 
from numpy.fft import irfft 

def fdaf(x, d, M, hop, mu=0.05, beta=0.9):
    """
    自适应滤波函数，支持可自定义的帧移长度（hop size）。

    参数：
    x -- 输入信号
    d -- 期望信号
    M -- 滤波器长度（每个数据块的长度）
    hop -- 帧移长度（每次处理的新样本数）
    mu -- 步长因子，控制滤波器系数更新的速度（默认值：0.05）
    beta -- 平滑因子，控制归一化因子的更新速度（默认值：0.9）

    返回：
    e -- 误差信号
    """
    # 初始化滤波器系数和归一化因子
    H = np.zeros(M + 1, dtype=complex)
    norm = np.full(M + 1, 1e-8)

    # 窗函数
    window = np.hanning(M)
    # 重叠缓存
    x_old = np.zeros(M)

    # 计算数据块的数量
    num_blocks = (len(x) - M) // hop + 1
    e = np.zeros(num_blocks * hop)

    for n in range(num_blocks):
        # 获取当前数据块
        start_idx = n * hop
        end_idx = start_idx + M
        x_n = np.concatenate([x_old, x[start_idx:end_idx]])
        d_n = d[start_idx:end_idx]
        # 更新重叠缓存
        x_old = x[start_idx:end_idx]

        # 频域处理
        X_n = rfft(x_n)
        y_n = irfft(H * X_n)[M:]
        e_n = d_n - y_n

        # 更新归一化因子
        e_fft = np.concatenate([np.zeros(M), e_n * window])
        E_n = rfft(e_fft)
        norm = beta * norm + (1 - beta) * np.abs(X_n) ** 2

        # 更新滤波器系数
        G = X_n.conj() * E_n / norm
        H += mu * G

        # 强制滤波器系数的后半部分为零
        h = irfft(H)
        h[M:] = 0
        H = rfft(h)

        # 存储误差信号
        e[start_idx:start_idx + hop] = e_n[:hop]

    return e

# ... 保持原有函数不变 ...

if __name__ == "__main__":
    print("此模块滤波结果存在错误")
    import numpy as np
    # 绘制结果
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'SimHei'  # 替换为你选择的字体
    # 生成测试信号
    fs = 1000
    t = np.arange(0, 2, 1/fs)
    s = 2*np.sin(2*np.pi*50*t) 
    x = 0.5*np.sin(2*np.pi*120*t)
    d = s + x
    
    # 运行滤波器（M=滤波器长度，hop=帧移）
    e = fdaf(d, x, M=512, hop=128, mu=0.05, beta=0.98)
    

    plt.figure(figsize=(12,6))
    
    # 时域对比
    plt.subplot(121)
    plt.plot(t[:500], d[:500], 'b', label='原始信号')
    plt.plot(t[:500], x[:500], 'r', alpha=0.6, label='干扰信号')
    plt.plot(t[:500], e[:500], 'g', linewidth=1.5, label='滤波结果')
    plt.title('时域对比 (前500点)')
    plt.legend()
    
    # 频域对比
    plt.subplot(122)
    freq = np.fft.rfftfreq(len(d), 1/fs)[:300]
    plt.plot(freq, np.abs(rfft(d))[:300], 'b', label='原始信号')
    plt.plot(freq, np.abs(rfft(x))[:300], 'r', alpha=0.6, label='参考信号')
    plt.plot(freq, np.abs(rfft(e))[:300], 'g', linewidth=1.5, label='滤波结果')
    plt.title('频域对比 (0-300Hz)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('fdaf_demo.png')
    plt.show()

