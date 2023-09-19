# '''Tạo tín hiệu âm thanh đơn điệu'''
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io.wavfile import write

# output_file = 'E:/Projects/LearnAI/Input/sample_audio.wav'  # Sửa đường dẫn file lưu âm thanh

# duration = 4  # in seconds
# frequency_sampling = 44100  # in Hz
# tone_freq = 784  # Sửa tên biến này từ "frequency_tone" thành "tone_freq"
# min_val = -4 * np.pi
# max_val = 4 * np.pi

# t = np.linspace(min_val, max_val, duration * frequency_sampling)
# audio_signal = np.sin(2 * np.pi * tone_freq * t)

# write(output_file, frequency_sampling, audio_signal)  # Sửa tên biến từ "signal_scaled" thành "audio_signal"

# audio_signal = audio_signal[:100]
# time_axis = 1000 * np.arange(0, len(audio_signal), 1) / float(frequency_sampling)  # Sửa tên biến "signal" thành "audio_signal", "sampling_freq" thành "frequency_sampling"

# plt.plot(time_axis, audio_signal, color='blue')
# plt.xlabel('Time in milliseconds')
# plt.ylabel('Amplitude')
# plt.title('Generated audio signal')
# plt.show()


# '''Tính năng trích xuất từ giọng nói'''
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io import wavfile
# from python_speech_features import mfcc, logfbank

# # Đọc tập tin âm thanh
# frequency_sampling, audio_signal = wavfile.read("E:/Projects/LearnAI/Input/sample_audio.wav")

# # Cắt tập tin âm thanh (chỉ lấy 15000 mẫu âm thanh)
# audio_signal = audio_signal[:15000]

# # Tính toán đặc trưng MFCC
# features_mfcc = mfcc(audio_signal, frequency_sampling)

# print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
# print('Length of each feature =', features_mfcc.shape[1])

# # Vẽ đồ thị MFCC
# plt.matshow(features_mfcc.T, cmap=plt.cm.jet)
# plt.title('MFCC')
# plt.colorbar()
# plt.show()

# # Tính toán đặc trưng Filter bank
# filterbank_features = logfbank(audio_signal, frequency_sampling)

# print('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
# print('Length of each feature =', filterbank_features.shape[1])

# # Vẽ đồ thị Filter bank
# plt.matshow(filterbank_features.T, cmap=plt.cm.jet)
# plt.title('Filter bank')
# plt.colorbar()
# plt.show()


# '''Nhận biết các từ đã nói'''
# import speech_recognition as sr

# # Tạo một đối tượng Recognizer để quản lý việc ghi âm và nhận dạng giọng nói
# recording = sr.Recognizer()

# # Mở microphone và điều chỉnh cho nhiễu xung quanh
# with sr.Microphone() as source:
#     recording.adjust_for_ambient_noise(source)
#     print("Please Say something:")
#     audio = recording.listen(source)

# try:
#     # Sử dụng Google Speech Recognition để nhận dạng văn bản từ âm thanh ghi âm
#     recognized_text = recording.recognize_google(audio)
#     print("You said: \n" + recognized_text)
# except Exception as e:
#     print(e)

