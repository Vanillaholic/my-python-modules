import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction
from scipy import signal
def generate_phase_encoded_signal(m_sequence, duration, f0, fs):
    t = cp.linspace(0, duration, int(duration * fs), endpoint=False)  # 时间轴
    # 初始化相位编码信号
    subpulse = cp.zeros_like(t)
    # 遍历 m-sequence，根据其值调整相位
    for i, bit in enumerate(m_sequence):
        start = int(i * len(t) / len(m_sequence))
        end = int((i + 1) * len(t) / len(m_sequence))
        if bit == 1:
            subpulse[start:end] = cp.cos(2 * cp.pi * f0 * t[start:end] + cp.pi)
        else:
            subpulse[start:end] = cp.cos(2 * cp.pi * f0 * t[start:end])
    template = subpulse[::-1]
    return subpulse , template




def range_doppler_map(tx_signal,rx_signal,fs,c,max_vel):
    '''
    tx_signal:发射的信号波形
    rx_signal:接收的信号波形
    fs:采样率
    c:波速
    max_vel:最大观测速度
    return:
        RDM:        range doppler map,一个横轴是速度,纵轴是距离的矩阵
        vel_axis:   速度轴,以便可视化
        range_axis: 距离轴,以便可视化
    '''
    #TODO根据发射的波形获取模板
    T = len(tx_signal)/fs             #脉冲长度
    sample_num = 3*len(tx_signal)     #设置RDM的samples数目为发射信号的三倍
    tx_paddle = np.concatenate([np.zeros(len(tx_signal)),tx_signal,np.zeros(len(tx_signal))])

    vel_resolution = 0.1              #速度bin
    range_resolution = c*T/2          #距离分辨率,由脉宽决定：delta r = c*T/2
    vel_axis = np.arange(-max_vel, max_vel + vel_resolution, vel_resolution)   #速度范围
    range_axis = np.linspace(0, range_resolution, sample_num)  # 生成数组

    #TODO根据发射波形获取不同的模板
    doppler_factor = 1+2*vel_axis/c   #多普勒因子
    pq_list = []                      #将多普勒因子变成分数，p是分母，q是分子
    for df in doppler_factor:
        f = Fraction(df).limit_denominator()
        # p, q 和 df 的对应需要注意是否和 resample_poly 一致
        pq_list.append((f.denominator, f.numerator))  # (p, q) = (denominator, numerator)

    templates =  np.zeros((len(vel_axis),sample_num))  #初始化模板
    for i, (vel, (p, q)) in enumerate(zip(vel_axis, pq_list)):
        # doppler_scaled_signal = resample_poly(rx_signal, q, p)
        # 先注意：函数 resample_poly(x, up, down) => up = q, down = p
        temp_resamp = signal.resample_poly(tx_paddle, q, p)

        # 如果得到的长度不是 sample_num，就截断或补零
        if len(temp_resamp) > sample_num:
            temp_resamp = temp_resamp[:sample_num]
        elif len(temp_resamp) < sample_num:
            temp_resamp = np.pad(temp_resamp, (0, sample_num - len(temp_resamp)), 'constant')
        # 存到第 i 行
        templates[i, :] = temp_resamp[::-1]

    #TODO利用不同的模板与接收信号进行匹配滤波，以便获取速度信息
    RDM =  np.zeros((sample_num,len(vel_axis)))  #初始化模板
    for i in range(len(vel_axis)):
        RDM[:,i] = signal.fftconvolve(templates[i, :], rx_signal, mode='same')
    RDM = abs(RDM[:])

    return RDM,vel_axis,range_axis


if __name__ == "__main__":

    # 参数配置
    fs = 4000           # 采样率
    duration = 0.1      # 单个子信号
    f0 = 1500            # 中心频率

    mseq1 = cp.asarray([0,0,0,0,1,0,0,1,0,1,1,0,0,1,1,1,1,1,0,0,0,1,1,0,1,1,1,0,1,0,1])
    mseq2 = cp.asarray([0,0,0,0,1,1,1,0,0,1,1,0,1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1])
    subpulse1,template = generate_phase_encoded_signal(mseq1,duration,f0,fs)
    subpulse2,  _      = generate_phase_encoded_signal(mseq2,duration,f0,fs)

    RDM, vel_axis, range_axis = range_doppler_map(cp.asnumpy(subpulse1), cp.asnumpy(subpulse1),max_vel=20, fs=fs,c=1500)
    plt.figure(figsize=(8, 6))
    plt.imshow(RDM, aspect='auto', extent=[vel_axis[0], vel_axis[-1], range_axis[0], range_axis[-1]])
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Range (m)")
    plt.title("Range Doppler Map")
    plt.colorbar() 


    #TODO绘制三维结果
    from mpl_toolkits.mplot3d import Axes3D  
    # 如果 RDM 的 shape 是 (N_range, N_vel)，那么 meshgrid 就要这样写：
    X, Y = np.meshgrid(vel_axis, range_axis)  # X对应速度，Y对应距离
    Z = RDM  # 这里就是你的二维阵列

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 画三维表面
    surf = ax.plot_surface(X, Y, Z,
                        cmap='viridis',  # 可以换成其他 colormap
                        linewidth=0,     # 网格线粗细，可根据需要设置
                        antialiased=True)

    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Range (m)")
    ax.set_zlabel("Amplitude")  # 或者根据你实际含义改

    fig.colorbar(surf, shrink=0.5, aspect=10)  # 右侧的 colorbar
    plt.title("3D Range-Doppler Map")
    plt.show()