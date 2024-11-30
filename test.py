import mmwave as mm
from mmwave.dataloader import DCA1000
from mmwave.dataloader import utils
import numpy as np
import matplotlib.pyplot as plt

adc_data = utils.parse_tsw1400("./mmwave_2/w1.bin", num_chirps_per_frame=17, num_frames=1000, num_ants= 2, num_adc_samples=256)

print(type(adc_data))
print(adc_data.shape)

radar_cube = mm.dsp.range_processing(adc_data[550])

print(type(radar_cube))
print(radar_cube.shape)

detMatrix = mm.dsp.doppler_processing(radar_cube=radar_cube,interleaved=False)

print(type(detMatrix))
print(detMatrix[0].shape)
print(detMatrix[1].shape)

range_resolution = mm.dsp.range_resolution(num_adc_samples=256,dig_out_sample_rate=3000,freq_slope_const=18.829)
print("range_resolution is {}m".format(range_resolution[0]))
doppler_resolution = mm.dsp.doppler_resolution(band_width=range_resolution[1] ,start_freq_const=77,ramp_end_time = 200,idle_time_const=100,num_loops_per_frame=17,num_tx_antennas=2)
print("doppler_resolution is {}m/s".format(doppler_resolution))

# 对 detMatrix[0]（Range-Doppler Map）执行频谱搬移
detMatrix_shifted = np.fft.fftshift(detMatrix[0], axes=1)
#可视化 detMatrix
plt.imshow(detMatrix[0], aspect='auto', cmap='hot', interpolation='nearest')
plt.title("Range-Doppler Map")
plt.xlabel("Doppler Bin")
plt.ylabel("Range Bin")
plt.colorbar(label="Amplitude")
plt.show()
