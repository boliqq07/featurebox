# -*- coding: utf-8 -*-

# @Time     : 2021/9/6 15:21
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx
import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号

# 采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
x = np.linspace(0, 1, 1400)

# 设置需要采样的信号，频率分量有200，400和600
y = 7 * np.sin(2 * np.pi * 200 * x) + 5 * np.sin(2 * np.pi * 400 * x) + 3 * np.sin(2 * np.pi * 600 * x)

plt.figure()
plt.plot(x, y)
plt.title('原始波形')

plt.figure()
plt.plot(x[0:50], y[0:50])
plt.title('原始部分波形（前50组样本）')
plt.show()

fft_y = fft(y)  # 快速傅里叶变换
print(len(fft_y))
print(fft_y[0:5])

N = 1400
x = np.arange(N)  # 频率个数
abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
angle_y = np.angle(fft_y)  # 取复数的角度

normalization_y = abs_y / N  # 归一化处理（双边频谱）
plt.figure()
plt.plot(x, normalization_y, 'g')
plt.title('双边频谱(归一化)', fontsize=9, color='green')
plt.show()
