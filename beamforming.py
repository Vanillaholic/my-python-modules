#import cupy.signal as csignal
import cupy as cp
import numpy as np
from scipy import signal

def CBF(DATA,d,min_freq_bin, max_freq_bin, Cspeed, fs, freq_domain=False):
    """
    NumPy实现的时延波束形成函数 (CPU版本)
    
    参数:
        DATA: 输入信号 (阵元数, 时域采样点数)
        min_freq_bin: 最小频率索引
        max_freq_bin: 最大频率索引
        Cspeed: 声速 (m/s)
        fs: 采样频率 (Hz)
        freq_domain: 是否返回频域波束形成结果
        
    返回:
        CBF_output_freq_domain: 波束形成后的频域数据 (如果freq_domain=True)
        CBF_output: 波束形成后的时域数据 (如果freq_domain=False)
    """
    # 获取信号参数
    M_arr, processSigLen = DATA.shape  # M_arr:阵元数  processSigLen:信号长度
    
    # 计算频率参数
    # df:频率分辨率              freq_range:频率范围  
    # min_freq_bin:最小频率索引  max_freq_bin:最大频率索引
    # num_freq_bins:频率bin数  
    df = fs / processSigLen                                    
    freq_range = np.arange(min_freq_bin, max_freq_bin + 1) * df 
    num_freq_bins = len(freq_range) 
    
    # 构建角度向量 (0°-180°)  Theta:角度向量  CosTheta:余弦值
    Theta = np.deg2rad(np.arange(180))  # 0到179度，共180个角度
    CosTheta = np.cos(Theta)
    
    # 构建阵元位置 (均匀线阵)  elementPosition:阵元位置
    elementPosition = np.arange(M_arr) * d  #d 阵元间距
    
    # 1. 对输入信号进行FFT  rf:频域信号
    rf = np.fft.fft(DATA, axis=1)
    
    # 初始化输出矩阵
    CBF_output_freq_domain = np.zeros((180, processSigLen), dtype=np.complex128)
    
    # 选择当前频率范围的信号
    rf_selected = rf[:, min_freq_bin:max_freq_bin+1]
    
    # 2. 波束形成核心计算   phi:角度索引  steering_vec:导向矢量  beamformed:波束形成后的信号
    # 为每个角度创建导向矢量并执行波束形成
    for phi in range(180):
        # 计算当前角度的导向矢量
        # 使用广播机制高效计算
        steering_vec = np.exp(-2j * np.pi * elementPosition[:, np.newaxis] * 
                              freq_range[np.newaxis, :] * CosTheta[phi] / Cspeed)
        
        # 波束形成 (阵元维度求和)
        beamformed = np.sum(steering_vec * rf_selected, axis=0)
        
        # 存储结果
        CBF_output_freq_domain[phi, min_freq_bin:max_freq_bin+1] = beamformed
    
    # 3. 逆FFT获取时域输出
    CBF_output = np.fft.ifft(CBF_output_freq_domain, axis=1)
    
    if freq_domain:
        return CBF_output_freq_domain
    else:
        return CBF_output

def CBF_gpu(DATA,d, min_freq_bin, max_freq_bin, Cspeed, fs,freq_domain=False):
    """
    CuPy实现的时延波束形成函数
    
    参数:
        DATA: 输入信号 (阵元数, 时域采样点数)
        min_freq_bin: 最小频率索引
        max_freq_bin: 最大频率索引
        Cspeed: 声速 (m/s)
        fs: 采样频率 (Hz)
        freq_domian: 是否频域波束成形成
        
    返回:
        CBF_output_freq_domain: 波束形成后的频域数据
        CBF_output: 波束形成后的时域数据
    """
    # 获取信号参数
    M_arr, processSigLen = DATA.shape  # M_arr:阵元数  processSigLen:信号长度
    
    # 计算频率参数
    # df:频率分辨率              freq_range:频率范围  
    # min_freq_bin:最小频率索引  max_freq_bin:最大频率索引
    # num_freq_bins:频率bin数  
    df = fs / processSigLen                                    
    freq_range = cp.arange(min_freq_bin, max_freq_bin + 1) * df 
    num_freq_bins = len(freq_range) 
    
    # 构建角度向量 (0°-180°)  Theta:角度向量  CosTheta:余弦值
    Theta = cp.deg2rad(cp.arange(180))  # 0到179度，共180个角度
    CosTheta = cp.cos(Theta)
    
    # 构建阵元位置 (均匀线阵)  elementPosition:阵元位置
    elementPosition = cp.arange(M_arr) * d  # d: 阵元间距
    
    # 1. 对输入信号进行FFT  rf:频域信号
    rf = cp.fft.fft(DATA, axis=1)
    
    # 初始化输出矩阵  DATA_step1:输出矩阵
    CBF_output_freq_domain = cp.zeros((180, processSigLen), dtype=cp.complex64)
    
    # 选择当前频率范围的信号
    rf_selected = rf[:, min_freq_bin:max_freq_bin+1]
    
    # 2. 波束形成核心计算   phi:角度索引  steering_vec:导向矢量  beamformed:波束形成后的信号
    # 为每个角度创建导向矢量并执行波束形成
    for phi in range(180):
        # 计算当前角度的导向矢量
        # 使用广播机制高效计算
        steering_vec = cp.exp(-2j * cp.pi * elementPosition[:, cp.newaxis] * 
                              freq_range[cp.newaxis, :] * CosTheta[phi] / Cspeed)
        
        # 波束形成 (阵元维度求和)
        beamformed = cp.sum(steering_vec * rf_selected, axis=0)
        
        # 存储结果
        CBF_output_freq_domain[phi, min_freq_bin:max_freq_bin+1] = beamformed
    
    # 3. 逆FFT获取时域输出
    CBF_output = cp.fft.ifft(CBF_output_freq_domain, axis=1)
    
    if freq_domain:
        return CBF_output_freq_domain
    else:
        return CBF_output
    
# 示例使用方式
if __name__ == "__main__":
    # 模拟参数设置
    Cspeed = 1500.0    # 声速 (m/s)
    fs = 10000          # 采样频率 (Hz) - 需要根据实际情况设置
    M_arr = 192         # 阵元数
    processSigLen = 1024 # 信号长度
    d = 0.1875            # 阵元间距
    
    # 计算频率索引范围 (2.9kHz-3.1kHz)
    df = fs / processSigLen
    min_freq_bin = int(2900 / df)
    max_freq_bin = int(3100 / df)
    
    # 生成随机输入数据 (模拟接收信号)
    DATA = cp.random.randn(M_arr, processSigLen) + 1j*cp.random.randn(M_arr, processSigLen)
    
    # 执行波束形成
    CBF_output= CBF_gpu(
        DATA, d, min_freq_bin, max_freq_bin, Cspeed, fs
    )
    
    # 将结果转移到CPU (如果需要)
    #CBF_output_freq_cpu = cp.asnumpy(CBF_output)
    CBF_output_cpu = cp.asnumpy(CBF_output)

    
    print("波束形成完成!")
    #print(f"频域输出形状: {CBF_output_freq_cpu.shape}")
    print(f"时域输出形状: {CBF_output_cpu.shape}")
