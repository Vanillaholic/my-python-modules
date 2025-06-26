import numpy as np

def m_seq(gen, init=None):
    """
    生成 m 序列 (最大长度序列)
    
    参数:
    gen : list or numpy.ndarray
        生成多项式系数向量 [1, c_{n-1}, ..., c_0]，最高次项系数必须为1
        例如 x^4 + x + 1 写成 [1, 0, 0, 1, 1]
    init : list or numpy.ndarray, optional
        n位初始状态向量 [a_{n-1}, ..., a_0]，长度为n，至少有1个1
        如果为None，则使用默认初始状态 [1, 1, ..., 1]
    
    返回:
    seq : numpy.ndarray
        生成的m序列，长度为 2^n - 1
    
    参考: Sarwate & Pursley (1980)
    """
    gen = np.array(gen)
    
    # 检查最高次项系数
    if gen[0] != 1:
        raise ValueError('Highest-degree coefficient must be 1.')
    
    # 提取tap位置，翻转顺序以与寄存器位对齐
    taps = np.flip(gen[1:])  # c_{0: n-1}，顺序与寄存器位对齐
    n = len(taps)  # 寄存器级数
    N = 2**n - 1   # 序列周期
    
    # 设置初始状态
    if init is None:
        reg = np.ones(n, dtype=int)  # 默认初始状态 [1, 1, ..., 1]
    else:
        reg = np.array(init, dtype=int).flatten()  # 强制成一维数组
        
        if len(reg) != n:
            raise ValueError(f'Initial state length must equal degree n = {n}.')
        
        if not np.any(reg):
            raise ValueError('Initial state must be non-zero.')
    
    # 生成m序列
    seq = np.zeros(N, dtype=int)
    
    for k in range(N):
        seq[k] = reg[0]  # 输出最高位
        
        # 计算反馈：选中的tap位进行XOR运算
        feedback = np.sum(reg & taps) % 2
        
        # 左移寄存器并注入反馈
        reg = np.concatenate([reg[1:], [feedback]])
    
    return seq


'''costas 部分'''

from typing import List, Tuple, Union

def is_prime(n: int) -> bool:
    """判断n是否为质数"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def prime_factors(n: int) -> List[int]:
    """求n的质因数分解"""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return list(set(factors))

def find_generators(p: int) -> List[int]:
    """找到模p的所有本原根(生成元)"""
    if not is_prime(p):
        return []
    
    phi = p - 1
    factors = prime_factors(phi)
    generators = []
    
    for g in range(2, p):
        is_generator = True
        for f in factors:
            if pow(g, phi // f, p) == 1:
                is_generator = False
                break
        if is_generator:
            generators.append(g)
    
    return generators

def costas_seq(N: int, num: int) -> Tuple[np.ndarray, List[str]]:
    """
    生成指定阶数 N 的 num 个 Costas 序列
    
    参数:
    N : int
        序列阶数（正整数，表示序列长度）
    num : int
        需要生成的序列个数
    
    返回:
    costas_list : numpy.ndarray
        大小为 [num, N] 的矩阵，每行是一个长度为 N 的 Costas 序列
    method_list : List[str]
        列表，长度为 num，每项为相应序列的构造方法名称
    """
    
    # 已知的各阶 Costas 序列总数（1<=N<=29）
    known_counts = [1, 2, 4, 12, 40, 116, 200, 444, 760, 2160,
                   4368, 7852, 12828, 17252, 19612, 21104,
                   18276, 15096, 10240, 6464, 3536, 2052,
                   872, 200, 88, 56, 204, 712, 164]
    
    if N <= len(known_counts) and num > known_counts[N-1]:
        raise ValueError(f'阶数 {N} 的 Costas 序列最多只有 {known_counts[N-1]} 个，无法生成 {num} 个序列。')
    
    sequences = []
    methods = []
    
    # 1. Welch 构造法：若 N+1 为质数，则生成 Welch-Costas 序列
    p = N + 1
    if is_prime(p):
        generators = find_generators(p)
        for g in generators:
            seq = np.zeros(N, dtype=int)
            for i in range(1, N + 1):
                seq[i-1] = pow(g, i, p)
            
            sequences.append(seq)
            methods.append('Welch')
            
            if len(sequences) >= num:
                return np.array(sequences), methods
    
    # 2. Golomb 构造法
    # (a) 若 N+2 = q 为质数
    q = N + 2
    if is_prime(q):
        generators = find_generators(q)
        if len(generators) >= 2:
            alpha = generators[0]
            beta = generators[1]
            
            n_array = q - 1
            costas_mat = np.zeros((n_array, n_array), dtype=bool)
            
            for i in range(1, n_array + 1):
                Ai = pow(alpha, i, q)
                for j in range(1, n_array + 1):
                    Bj = pow(beta, j, q)
                    if (Ai + Bj) % q == 1:
                        costas_mat[i-1, j-1] = True
            
            # 验证每行每列都只有一个True
            if np.all(np.sum(costas_mat, axis=1) == 1) and np.all(np.sum(costas_mat, axis=0) == 1):
                seq = np.zeros(N, dtype=int)
                for i in range(N):
                    j = np.where(costas_mat[i, :])[0]
                    if len(j) > 0:
                        seq[i] = j[0] + 1  # MATLAB索引从1开始
                
                # 检查是否已存在相同序列
                is_duplicate = any(np.array_equal(seq, existing) for existing in sequences)
                if not is_duplicate:
                    sequences.append(seq)
                    methods.append('Golomb')
                    
                    if len(sequences) >= num:
                        return np.array(sequences), methods
    
    # (b) 若 N+3 = q 为质数
    q = N + 3
    if is_prime(q):
        generators = find_generators(q)
        alpha, beta = None, None
        
        # 寻找满足 α + β = 1 的本原元对
        for a in generators:
            b = (1 - a) % q
            if b in generators and b != a:
                alpha, beta = a, b
                break
        
        # 特殊情况：α = β = (1/2) mod q
        if alpha is None:
            for a in generators:
                if (2 * a) % q == 1:
                    alpha = beta = a
                    break
        
        if alpha is not None:
            big_n = q - 1
            big_mat = np.zeros((big_n, big_n), dtype=bool)
            
            for i in range(1, big_n + 1):
                Ai = pow(alpha, i, q)
                for j in range(1, big_n + 1):
                    Bj = pow(beta, j, q)
                    if (Ai + Bj) % q == 1:
                        big_mat[i-1, j-1] = True
            
            # 删除第一行和第一列
            sub_mat = big_mat[1:, 1:]
            
            if (sub_mat.shape[0] == N and 
                np.all(np.sum(sub_mat, axis=1) == 1) and 
                np.all(np.sum(sub_mat, axis=0) == 1)):
                
                seq = np.zeros(N, dtype=int)
                for i in range(N):
                    j = np.where(sub_mat[i, :])[0]
                    if len(j) > 0:
                        seq[i] = j[0] + 1
                
                is_duplicate = any(np.array_equal(seq, existing) for existing in sequences)
                if not is_duplicate:
                    sequences.append(seq)
                    methods.append('Golomb')
                    
                    if len(sequences) >= num:
                        return np.array(sequences), methods
    
    # 3. 穷举搜索法
    if len(sequences) < num:
        if N > 10:  # 控制计算复杂度
            raise ValueError(f'无法通过已知方法构造 {num} 个阶数为 {N} 的 Costas 序列。')
        
        exhaustive_seqs = exhaustive_search(N, num - len(sequences), sequences)
        
        for seq in exhaustive_seqs:
            sequences.append(seq)
            methods.append('Exhaustive')
        
        if len(sequences) < num:
            raise ValueError(f'无法构造 {num} 个互异的阶 {N} Costas 序列（合法序列总数不足）。')
    
    return np.array(sequences), methods

def exhaustive_search(N: int, needed: int, existing_seqs: List[np.ndarray]) -> List[np.ndarray]:
    """穷举搜索Costas序列"""
    found_seqs = []
    seq = np.zeros(N, dtype=int)
    used = np.zeros(N, dtype=bool)
    used_diff = np.zeros((N-1, 2*N-1), dtype=bool)
    offset = N - 1  # 列差偏移量
    
    def backtrack(pos: int) -> bool:
        if pos >= N:
            # 完成一个序列
            new_seq = seq.copy()
            
            # 确保序列未收录过
            is_duplicate = (any(np.array_equal(new_seq, existing) for existing in existing_seqs) or
                          any(np.array_equal(new_seq, existing) for existing in found_seqs))
            
            if not is_duplicate:
                found_seqs.append(new_seq)
            
            return len(found_seqs) < needed
        
        for val in range(1, N + 1):
            if not used[val - 1]:
                # 检查与之前元素形成的向量是否唯一
                unique_diff = True
                for prev in range(pos):
                    dr = pos - prev
                    dc = val - seq[prev]
                    if used_diff[dr - 1, dc + offset]:
                        unique_diff = False
                        break
                
                if not unique_diff:
                    continue
                
                # 选择当前值
                seq[pos] = val
                used[val - 1] = True
                
                # 标记新产生的差分
                for prev in range(pos):
                    dr = pos - prev
                    dc = seq[pos] - seq[prev]
                    used_diff[dr - 1, dc + offset] = True
                
                # 递归
                cont = backtrack(pos + 1)
                
                # 回溯
                for prev in range(pos):
                    dr = pos - prev
                    dc = seq[pos] - seq[prev]
                    used_diff[dr - 1, dc + offset] = False
                
                used[val - 1] = False
                seq[pos] = 0
                
                if not cont or len(found_seqs) >= needed:
                    return False
        
        return True
    
    backtrack(0)
    return found_seqs

# 测试用例
if __name__ == "__main__":


    # 测试生成M序列
    print("=========测试生成M序列=================\n")
    gen_435 = [1, 0, 0, 0, 1, 1, 1, 0, 1]  # x^8 + x^4 + x^3 + x^2 + 1
    init_state = [1, 0, 1, 0, 1, 0, 1, 0]   # 8位初始状态，必须非零
    
    # 生成m序列
    M_array = m_seq(gen_435, init_state)
    
    print(f"生成多项式: {gen_435}")
    print(f"初始状态: {init_state}")
    print(f"序列长度: {len(M_array)}")
    print(f"前20位: {M_array[:20]}")
    print(f"后20位: {M_array[-20:]}")
    
    # 验证周期性
    M_array_extended = m_seq(gen_435, init_state)
    period_check = np.array_equal(M_array[:50], 
                                 np.tile(M_array, 2)[:50])
    print(f"周期性验证: {period_check}")
    
    # 测试默认初始状态
    print("\n使用默认初始状态:")
    M_default = m_seq(gen_435)
    print(f"前20位: {M_default[:20]}")





    # 测试生成Costas序列
    print("=========测试生成costas序列=================\n")
    try:
        N = 18
        num = 3
        costas_list, method_list = costas_seq(N, num)
        
        print(f"生成 {num} 个阶数为 {N} 的 Costas 序列:")
        print(f"序列矩阵形状: {costas_list.shape}")
        
        for i, (seq, method) in enumerate(zip(costas_list, method_list)):
            print(f"序列 {i+1}: {seq} (方法: {method})")
        
        # 验证Costas性质
        def verify_costas(seq):
            N = len(seq)
            diff_vectors = set()
            for i in range(N):
                for j in range(i+1, N):
                    diff_vec = (j - i, seq[j] - seq[i])
                    if diff_vec in diff_vectors:
                        return False
                    diff_vectors.add(diff_vec)
            return True
        
        print("\nCostas性质验证:")
        for i, seq in enumerate(costas_list):
            is_valid = verify_costas(seq)
            print(f"序列 {i+1}: {'✓' if is_valid else '✗'}")
            
    except ValueError as e:
        print(f"错误: {e}")