import mmwave as mm
from mmwave.dataloader import DCA1000
from mmwave.dataloader import utils
import numpy as np
import matplotlib.pyplot as plt

static_data = utils.parse_tsw1400("./mmwave_3/static.bin", num_chirps_per_frame=34, num_frames=1000, num_ants= 2, num_adc_samples=598)
adc_data = utils.parse_tsw1400("./mm4/w1.bin", num_chirps_per_frame=34, num_frames=1000, num_ants= 2, num_adc_samples=598)


def cal2dFFT0():
    radar_cube = mm.dsp.range_processing(static_data[600])
    detMatrix = mm.dsp.doppler_processing(radar_cube=radar_cube,interleaved=False)
    return detMatrix

def cal2dFFTByFrame(index):
    radar_cube = mm.dsp.range_processing(adc_data[index])

    detMatrix = mm.dsp.doppler_processing(radar_cube=radar_cube,interleaved=False)
    return detMatrix

detMatrix_0 = cal2dFFT0()
twod_fft_0 = detMatrix_0[0]
detMatrix = cal2dFFTByFrame(200)
twod_fft = detMatrix[0]

twod_fft = twod_fft-twod_fft_0

range_resolution = mm.dsp.range_resolution(num_adc_samples=598,dig_out_sample_rate=3000,freq_slope_const=18.829)
print("range_resolution is {}m".format(range_resolution[0]))
doppler_resolution =3*10e8/(2*77*10e3*17*300)
print("doppler_resolution is {}m/s".format(doppler_resolution))

# 计算距离轴
num_range_bins = twod_fft.shape[0]  # Range Bin 数量
print(num_range_bins)
range_axis = np.arange(num_range_bins) * range_resolution[0]  # 实际距离轴

# 计算速度轴
num_doppler_bins = twod_fft.shape[1]  # Doppler Bin 数量
print(num_doppler_bins)
doppler_bins = np.arange(-num_doppler_bins//2+1, num_doppler_bins//2+1)  # 搬移后的 Doppler Bin 索引
print(np.arange(-num_doppler_bins//2, num_doppler_bins//2))
print(doppler_bins)
velocity_axis = doppler_bins * doppler_resolution  # 实际速度轴
print(velocity_axis)

# 搬移 Doppler 频谱
detMatrix_shifted = np.fft.fftshift(twod_fft, axes=1)
threshold_dBFS = 10  # 阈值 9 dBFS

# 应用阈值
detMatrix_thresholded = np.where(detMatrix_shifted >= threshold_dBFS, detMatrix_shifted, 0)

# 可视化带物理意义的 Range-Doppler Map
plt.imshow(
    detMatrix_thresholded,
    aspect='auto',
    cmap='coolwarm',
    interpolation='nearest',
    extent=[velocity_axis[0], velocity_axis[-1], range_axis[-1], range_axis[0]]
)
plt.title("Range-Doppler Map with Physical Units")
plt.xlabel("Velocity (m/s)")
plt.ylabel("Range (m)")
plt.colorbar(label="Amplitude")
plt.show()