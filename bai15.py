'''Hình dung tín hiệu âm thanh'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile # Dùng wavfile để đọc file âm thanh wav

# Đọc tệp âm thanh
frequency_sampling, audio_signal = wavfile.read('E:\Projects\LearnAI\Input\sample_audio.wav') # Lưu tần số mẫu frequency_sampling và tín hiệu âm thanh audio_signal 

print('\nSignal shape:', audio_signal.shape) # Print shape của tín hiệu âm thanh, số mẫu âm thanh trong file
print('Signal Datatype:', audio_signal.dtype) # Print type data của tín hiệu âm thanh
print('Signal duration:', round(audio_signal.shape[0] / float(frequency_sampling), 2), 'seconds') # Tính và in ra thời gian hoàn thành của tín hiệu âm thanh dựa trên số mẫu và tần số lấy mẫu

# Chuẩn hóa tín hiệu
audio_signal = audio_signal / np.power(2, 15) # Chuyển đổi tín hiệu từ kiểu số nguyên có phạm vi từ -32768 đến 32767 thành kiểu số thực nằm trong khoảng từ -1 đến 1.

# Chọn một phần của tín hiệu để hiển thị (100 mẫu đầu tiên)
audio_signal = audio_signal[:100] # Lấy 100 mẫu đầu tiên
time_axis = 1000 * np.arange(0, len(audio_signal), 1) / float(frequency_sampling) # Tạo trục time thể hiện tín hiệu theo mili giây, tính toán thời gian tương ứng cho mỗi mẫu âm thanh

# Vẽ biểu đồ
plt.plot(time_axis, audio_signal, color='blue') # Vẽ biểu đồ trên trục thời gian
plt.xlabel('Time (milliseconds)') # Đặt label cho trục x 
plt.ylabel('Amplitude') # Đặt label cho trục y
plt.title('Input audio signal') # Đặt tiêu đề
plt.show() # Show


'''Đặc trưng cho tín hiệu âm thanh'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

frequency_sampling, audio_signal = wavfile.read('E:\Projects\LearnAI\Input\sample_audio.wav')

print('\nSignal shape:', audio_signal.shape)
print('Signal Datatype:', audio_signal.dtype)
print('Signal duration:', round(audio_signal.shape[0] / float(frequency_sampling), 2), 'seconds')

audio_signal = audio_signal / np.power(2, 15)

length_signal = len(audio_signal) # Tính chiều dài (số mẫu) của tín hiệu âm thanh
half_length = np.ceil((length_signal + 1) / 2.0).astype(int) # Tính giá trị half_length cho việc tính Fast Fourier Transform

signal_frequency = np.fft.fft(audio_signal) # Biến đổi FFT của tín hiệu âm thanh và lưu vào signal_frequency

signal_frequency = abs(signal_frequency[0:half_length]) / length_signal # Tính biên độ của phổ tần số bằng cách lấy giá trị tuyệt đối chia cho length_signal
signal_frequency **= 2 # Tính công suất tín hiệu = bình phương biên độ phổ

len_fts = len(signal_frequency) # TÍnh chiều dài của phổ tần

if length_signal % 2: # Kiểm tra phải số lẻ không. Nếu lẻ -> nhân đôi giá trị từ vị trí 1-lenfts
    signal_frequency[1:len_fts] *= 2
else:
    signal_frequency[1:len_fts-1] *= 2
    
signal_power = 10 * np.log10(signal_frequency) # Tính công suất tín hiệu (dB) 

x_axis = np.arange(0, half_length, 1) * (frequency_sampling / length_signal) / 1000.0 # Tạo trục x cho biểu đồ phổ tần, tính toán F tương ứng với từng mấu FFT và chuyển thành kHz

plt.figure() # Tạo hình vẽ mới
plt.plot(x_axis, signal_power, color='black') # Vẽ biểu đồ 
plt.xlabel('Frequency (kHz)') # Đặt label cho trục x
plt.ylabel('Signal power (dB)') # Đặt label cho trục y
plt.show() # Show