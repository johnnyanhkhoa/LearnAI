import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# SVM
import pandas as pd
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn import svm

# Import Scikit-learn dataset
data = load_breast_cancer()

label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
# print(labels) # in ra các giá trị binary, 0 là ung thư ác tính và 1 là lành tính
# print(feature_names) # in ra các đặc trưng của bộ dữ liệu trên
# print(features) # in ra các giá trị tương ứng với các đặc trưng

# Tổ chức dữ liệu thành tập hợp
train, test, train_labels, test_labels = train_test_split(features,labels,test_size = 0.40, random_state = 42)

# Xây dựng model
gnb = GaussianNB()
model = gnb.fit(train, train_labels) # Train cho tập dữ liệu train với nhãn train_lables tương ứng qua phương thức fit

# Đánh giá mô hình và độ chính xác của mô hình
preds = gnb.predict(test) # Dự đoán label của các mẫu data trong test
# print(preds)
# print(accuracy_score(test_labels,preds)) # Tính toán độ chính xác của mô hình dự đoán. Nhãn thực tế là test_labels và preds là nhãn được dự đoán


'''Xây dựng Classifier bằng python'''
# SVM (Support Vector Machines)
iris = datasets.load_iris() # Load bộ data Iris (thông tin về các đoạn đài hoa và cánh hoa của 3 loài hoa Iris khác nhau)
x = iris.data[:, :2] # Tạo ra ma trận các mẫu dữ liệu, ở đây lấy 2 đặc trưng đầu tiên là đoạn đài hoa và cánh hoa
y = iris.target # Tạo label (ở đây là loại hoa) tương ứng với mỗi mẫu data trong dataset

x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1 # Tính giá trị min max để xác định phạm vi trục x trên đồ thị
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1 # Tính giá trị min max để xác định phạm vi trụ y trên đồ thị
h = (x_max / x_min)/100 # Tính bước nhảy của lưới trên đồ thị dựa trên phạm vi trục x và y. Giá trị 100 để điều chỉnh mật độ lưới
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), # Tạo lưới các điểm dữ liệu trên đồ thị bằng hàm meshgrid. Tạo ra các mảng xx và yy chứa tất cả các điểm trên lưới
np.arange(y_min, y_max, h))
x_plot = np.c_[xx.ravel(), yy.ravel()] # Tạo ra matrix mới 'x_plot' bằng cách "làm phẳng" các mảng xx và yy. Giúp sử dụng ma trận mới để dự đoán và trực quan hóa biên quyết định trên đồ thị

c = 1.0 # Train mô hình SVM với kernel tuyến tính và tham số c = 10. Tham số C quy định mức độ ưu tiên giữa việc tìm ra biên quyết định chính xác và việc chấp nhận sai sót phân loại.

svc_classifier = svm.SVC(kernel='linear', C=c, decision_function_shape='ovr') # Tạo var(đối tượng SVM) tới kernel tuyến tính và tham số C. decision_function_shape cho phép phân loại đa lớp thông qua One-vs-Rest (OvR)
svc_classifier.fit(x, y) # Sử dụng fit() để train mô hình trên data đặc trưng x và label tương ứng y

z = svc_classifier.predict(x_plot) # Sử dụng mô hình đã được huấn luyện (svc_classifier) để dự đoán label cho các điểm data trong x_plot
z = z.reshape(xx.shape) # Thay đổi hình dạng của z để phù hợp với hình dạng của lưới xx.
plt.figure(figsize = (15, 5)) # Tạo ra một hình vẽ với kích thước (15, 5) để chứa các đồ thị.
plt.subplot(121) # Tạo ra một khu vực đồ thị 1x2 và chọn vị trí thứ nhất để vẽ đồ thị.
plt.contourf(xx, yy, z, cmap = plt.cm.tab10, alpha = 0.3) #  Vẽ các đường đồng mức của biên quyết định trên đồ thị. Các đường đồng mức được tạo từ xx, yy và z. Các mức độ màu sắc của các đường đồng mức được chọn từ bản đồ màu plt.cm.tab10 và có độ trong suốt alpha=0.3.
plt.scatter(x[:, 0], x[:, 1], c = y, cmap = plt.cm.Set1) # Vẽ các điểm data trên đồ thị, màu sắc của các điểm đc xác định bởi label y và dựa trên bản đồ màu plt.cm.Set1
plt.xlabel('Sepal length') # Thiết lập label trục x
plt.ylabel('Sepal width') # Thiết lập label trục y
plt.xlim(xx.min(), xx.max()) # Thiết lập limit trục x dự trên min max của xx
plt.title('SVC with linear kernel') # Đặt tiêu đề cho đồ thị