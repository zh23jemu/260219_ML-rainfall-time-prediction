import numpy as np
import pandas as pd
import os

def generate_sequence_data(length=240, noise_level=0.5, seed=1000):
    """
    生成时间序列数据
    
    参数:
    - length: 数据长度
    - noise_level: 噪声水平
    - seed: 随机种子
    
    返回:
    - 生成的数据
    """
    np.random.seed(seed)
    
    # 基础序列
    t = np.linspace(0, 4 * np.pi, length)
    trend = 0.1 * t
    seasonal = 5 * np.sin(t)
    noise = np.random.normal(0, noise_level, length)
    
    # 组合成时间序列
    series = trend + seasonal + noise
    
    return series

def create_battery_data(base_series, num_batteries=4, variation=0.2, seed=1000):
    """
    为不同电池创建略有不同的数据
    
    参数:
    - base_series: 基础序列
    - num_batteries: 电池数量
    - variation: 变异程度
    - seed: 随机种子
    
    返回:
    - 电池数据字典
    """
    np.random.seed(seed)
    battery_data = {}
    
    battery_names = ['B0005', 'B0006', 'B0007', 'B0018']
    
    for i in range(num_batteries):
        # 添加一些随机变异
        battery_variation = np.random.normal(0, variation, len(base_series))
        battery_series = base_series + i * variation + battery_variation
        
        # 确保数据为正
        battery_series = np.maximum(battery_series, 0.1)
        
        battery_data[battery_names[i]] = battery_series
    
    return battery_data

def save_data(data_dir='./data'):
    """
    生成并保存数据
    
    参数:
    - data_dir: 数据保存目录
    """
    # 创建数据目录
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 生成基础序列
    base_series = generate_sequence_data()
    
    # 创建电池数据
    battery_data = create_battery_data(base_series)
    
    # 保存基础数据
    pd.DataFrame(base_series).to_csv(os.path.join(data_dir, 'data.csv'), index=False, header=['value'])
    print(f"基础数据已保存到 {os.path.join(data_dir, 'data.csv')}")
    
    # 保存电池数据
    for battery_name, series in battery_data.items():
        pd.DataFrame(series).to_csv(os.path.join(data_dir, f'{battery_name}.csv'), index=False, header=['value'])
        print(f"电池数据已保存到 {os.path.join(data_dir, f'{battery_name}.csv')}")
    
    return base_series, battery_data

if __name__ == "__main__":
    base_series, battery_data = save_data()
    print("数据生成完成!")
