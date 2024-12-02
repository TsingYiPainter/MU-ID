import mmwave as mm
from mmwave.dataloader import DCA1000
from mmwave.dataloader import utils
import numpy as np
import matplotlib.pyplot as plt

adc_data = utils.parse_tsw1400("./mmwave_2/y1.bin", num_chirps_per_frame=17, num_frames=1000, num_ants= 2, num_adc_samples=256)

num_frames = adc_data.shape[0]  # 总帧数
num_range_bins = adc_data.shape[-1]  # Range Bin 数量
num_doppler_bins = adc_data.shape[1] # Doppler Bin 数量
print(num_frames)
print(num_range_bins)
print(num_doppler_bins)


def cal2dFFT0():
    radar_cube = mm.dsp.range_processing(adc_data[0])
    detMatrix = mm.dsp.doppler_processing(radar_cube=radar_cube,interleaved=False)
    return detMatrix[0]

def cal2dFFTByFrame(index,origin2dFFT):
    radar_cube = mm.dsp.range_processing(adc_data[index])
    detMatrix = mm.dsp.doppler_processing(radar_cube=radar_cube,interleaved=False)
    twod_fft = detMatrix[0]

    twod_fft = twod_fft-origin2dFFT
    # 搬移 Doppler 频谱
    detMatrix_shifted = np.fft.fftshift(twod_fft, axes=1)

    # 定义满量程幅值和阈值
    # full_scale = np.max(detMatrix_shifted)  # 假设最大值为满量程
    threshold_dBFS = 9  # 阈值 9 dBFS

    # # 计算幅度和 dBFS 值
    # amplitude = np.abs(detMatrix_shifted)  # 计算幅度
    # dBFS = 20 * np.log10(amplitude / full_scale)  # 计算 dBFS

    # 应用阈值
    detMatrix_thresholded = np.where(detMatrix_shifted >= threshold_dBFS, detMatrix_shifted, 0)
    return detMatrix_thresholded

range_resolution = mm.dsp.range_resolution(num_adc_samples=256,dig_out_sample_rate=3000,freq_slope_const=18.829)
#print("range_resolution is {}m".format(range_resolution[0]))
doppler_resolution = mm.dsp.doppler_resolution(band_width=range_resolution[1] ,start_freq_const=77,ramp_end_time = 200,idle_time_const=100,num_loops_per_frame=17,num_tx_antennas=2)
#print("doppler_resolution is {}m/s".format(doppler_resolution))
doppler_bins = np.arange(-num_doppler_bins//2+1, num_doppler_bins//2+1)  # 搬移后的 Doppler Bin 索引
velocity_axis = doppler_bins * doppler_resolution  # 实际速度轴



# 初始化速度矩阵
velocity_map = np.zeros((num_range_bins, num_frames))
origin2dFFT = cal2dFFT0()
# 逐帧处理
for frame_idx in range(num_frames):
    detMatrix_shifted = cal2dFFTByFrame(frame_idx,origin2dFFT)
    R_hat = detMatrix_shifted / np.max(detMatrix_shifted, axis=1, keepdims=True)
    # 计算每个 Range Bin 的主导速度 V̂_i
    for i in range(num_range_bins):
        velocity_map[i, frame_idx] = np.sum(R_hat[i, :] * velocity_axis) / num_doppler_bins


# 绘制最终热力图
range_axis = np.arange(num_range_bins) * range_resolution[0]
plt.imshow(
    velocity_map,
    aspect='auto',
    cmap='jet',
    interpolation='nearest',
    extent=[0, num_frames, range_axis[-1], range_axis[0]]
)
plt.title("Range vs Frame Index (Dominant Velocity Heatmap)")
plt.xlabel("Frame Index")
plt.ylabel("Range (m)")
plt.colorbar(label="Dominant Velocity (m/s)")
plt.show()

# 计算距离轴
# num_range_bins = twod_fft.shape[0]  # Range Bin 数量
# print(num_range_bins)
# range_axis = np.arange(num_range_bins) * range_resolution[0]  # 实际距离轴

# # 计算速度轴
# num_doppler_bins = twod_fft.shape[1]  # Doppler Bin 数量
# print(num_doppler_bins)


# # 可视化带物理意义的 Range-Doppler Map
# plt.imshow(
#     detMatrix_thresholded,
#     aspect='auto',
#     cmap='coolwarm',
#     interpolation='nearest',
#     extent=[velocity_axis[0], velocity_axis[-1], range_axis[-1], range_axis[0]]
# )
# plt.title("Range-Doppler Map with Physical Units")
# plt.xlabel("Velocity (m/s)")
# plt.ylabel("Range (m)")
# plt.colorbar(label="Amplitude")
# plt.show()
