'''Logistic Regression'''

import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# Định nghĩa matrix X, row đại diện cho mỗi mẫu data, column đại diện cho đặc trưng của mẫu data
X = np.array([[2, 4.8], 
              [2.9, 4.7],
              [2.5, 5],
              [3.2, 5.5],
              [6, 5],
              [7.6, 4],
              [3.2, 0.9],
              [2.9, 1.9],
              [2.4, 3.5],
              [0.5, 3.4],
              [1, 4],
              [0.9, 5.9]])

y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]) # Định nghĩa vector y chưa label tương ứng với mỗi mẫu data trong matrix X

Classifier_LR = linear_model.LogisticRegression(solver = 'liblinear', C = 75) # Tạo đối tượng LogisticRegression với các tham số: solver = 'liblinear' chỉ định phương pháp giải quyết tối ưu hóa trong quá trình training, ở đây sử dụng liblinear (1 thuật toán tối ưu và phổ biến cho Logistic Regression) ; C=75 là tham số điều chỉnh regularization strength trong mô hình Logistich Regression, giá trị C càng cao thì mô hình sẽ càng kh có đàn hồi nhiều dẫn đến overfitting

Classifier_LR.fit(X, y) # Sử dụng fit() để train mô hình Logistich Regression với data X và label tương ứng y

def Logistic_visualize(Classifier_LR, X, y): # Tạo hàm dể trực quan hóa mô hình
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0 # Tính max min trục x để xác định phạm vi biểu đồ
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0 # Tính max min trục y để xác định phạm vi biểu đồ

    mesh_step_size = 0.02 # Xác định size bước giữa các point trên lưới

    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), # Tạo lưới điểm để dự đoán và trực quan hóa bề mặt quyết định của mô hình.
                    np.arange(min_y, max_y, mesh_step_size)) 

    output = Classifier_LR.predict(np.c_[x_vals.ravel(), y_vals.ravel()]) # Dự đoán label cho các point trên lưới bằng mô hình Logistich Regression đã train
    output = output.reshape(x_vals.shape) # Định dạng output để có cùng size với lưới point
    plt.figure()
    plt.pcolormesh(x_vals, y_vals, output, cmap = plt.cm.gray) # Vẽ bề mặt quyết định của mô hình bằng các mô hình bằng các màu sắc trên lưới điểm
    
    plt.scatter(X[:, 0], X[:, 1], c = y, s = 75, edgecolors = 'black',  # Vẽ các điểm data từ tập X trên biểu đồ, mỗi điểm có màu sắc tương ứng với label y
    linewidth=1, cmap = plt.cm.Paired)

    plt.xlim(x_vals.min(), x_vals.max()) # Đặt giới hạn trục x và y của biểu đồ để bao quát các điểm dữ liệu và bề mặt quyết định.
    plt.ylim(y_vals.min(), y_vals.max()) # Đặt giới hạn trục x và y của biểu đồ để bao quát các điểm dữ liệu và bề mặt quyết định.
    plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0))) # Đặt các vạch chia trên trục x và y để tăng độ dễ nhìn của biểu đồ.
    plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0))) # Đặt các vạch chia trên trục x và y để tăng độ dễ nhìn của biểu đồ.
    plt.show()

print(Logistic_visualize(Classifier_LR,X,y))