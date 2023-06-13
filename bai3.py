# Nhập các useful packages
import numpy as np
import sklearn.preprocessing
from sklearn import preprocessing



'''Kỹ thuật tiền xử lý dữ liệu'''

# Raw data
input_data = np.array([[2.1, -1.9, 5.5],
                      [-1.5, 2.4, 3.5],
                      [0.5, -7.9, 5.6],
                      [5.9, 2.3, -5.8]])


# Binarization
data_binarized = sklearn.preprocessing.Binarizer(threshold = 0.5).transform(input_data)
print("\nBinarized data:\n", data_binarized)


# Mean Removal
print("Mean = ", input_data.mean(axis = 0)) # Trung bình của mỗi cột (axis = 0 là cột dọc, axis = 1 là hàng ngang)
print("Std deviation = ", input_data.std(axis = 0)) # Tính độ lệch chuẩn của mỗi cột (= căn bậc hai(phương sai))

data_scaled = sklearn.preprocessing.scale(input_data) # scale() là phương pháp chuẩn hóa Z-score. Ở đây sẽ tạo ra 1 ma trận mới với mỗi giá trị trong mảng = (giá trị cũ - mean) / std 
print(data_scaled)
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis = 0))


# Scaling 
data_scaler_minmax = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print ("\nMin max scaled data:\n", data_scaled_minmax)


# Normalization 
# L1
data_normalized_l1 = sklearn.preprocessing.normalize(input_data, norm = 'l1')
print("\nL1 normalized data:\n", data_normalized_l1)

# L2
data_normalized_l2 = sklearn.preprocessing.normalize(input_data, norm = 'l2')
print("\nL2 normalized data:\n", data_normalized_l2)



'''Label trong dữ liệu'''


# Sample input labels
input_labels = ['red','black','red','green','black','yellow','white']

# Creating the label encoder
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

# encoding a set of labels
test_labels = ['green','red','black']
encoded_values = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))

# decoding a set of values
encoded_values = [3,0,4,1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values =", encoded_values)
print("\nDecoded labels =", list(decoded_list))
