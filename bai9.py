# # Tính toán điểm Silhouette
# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs
# from sklearn import metrics

# X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.40, random_state=0) # Tạo tập data với make_blobs. X là mảng 2D chứa các điểm data, y_true là label

# scores = [] # Tạo empty list để lưu điểm Silhouette
# values = np.arange(2, 10) # Tạo mảng numpy chứa value 2 - 9 

# for num_clusters in values:
#     kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10) # Tạo đối tượng KMeans với số cụm num_clusters và sử dụng phương pháp k-means++ để chọn các điểm tâm
#     kmeans.fit(X) # Thực hiện K-means trên data X

#     score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean', sample_size=len(X)) # Tính điểm Silhouette dựa trên kết quả phân clusters KMeans. Tham số metric='euclidean' xác định phương thức tính khoảng cách là khoảng cách Euclidean. Tham số sample_size=len(X) chỉ định kích thước mẫu được sử dụng để tính điểm Silhouette.

#     print("\nNumber of clusters =", num_clusters)
#     print("Silhouette score =", score)
#     scores.append(score)

# num_clusters = np.argmax(scores) + values[0] # Tím số lượng cluster tối ưu bằng cách lấy Silhoutte max cộng values[0] để có số lượng clusters final.
# print('\nOptimal number of clusters =', num_clusters)


# # Thuật toán Finding Nearest Neighbors
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.neighbors import NearestNeighbors

# A = np.array([[3.1, 2.3], [2.3, 4.2], [3.9, 3.5], [3.7, 6.4], [4.8, 1.9], 
#              [8.3, 3.1], [5.2, 7.5], [4.8, 4.7], [3.5, 5.1], [4.4, 2.9],]) # Tạo array numpy A, mỗi hàng đại điện cho một điểm data 2D (2 thuộc tính)

# k = 3 # Số điểm gần nhất cần tìm cho điểm data test

# test_data = [3.3, 2.9] # Điểm data test để tìm các điểm gần nhất

# plt.figure()
# plt.title('Input data')
# plt.scatter(A[:,0], A[:,1], marker='o', s=100, color='black') # Vẽ các điểm data từ A. Các điểm được ký hiệu bằng hình tròn (marker='0), kích thước 100 và có màu đen 

# knn_model = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(A) # Tạo đối tượng knn_model là mô hình NearestNeighbors và khớp với data mẫu A với số điểm gần nhất k 
# distances, indices = knn_model.kneighbors([test_data]) # Tìm các điểm gần nhất với test_data, sử dụng knn_model, distance từ test_data đến các điểm gần nhất và indicies chứa chỉ số các điểm gần nhất trong A

# print("\nK Nearest Neighbors:")
# for rank, index in enumerate(indices[0][:k], start=1):
#     print(str(rank) + " is", A[index])

# plt.figure()
# plt.title('Nearest neighbors')
# plt.scatter(A[:, 0], A[:, 1], marker='o', s=100, color='k')
# plt.scatter(A[indices][0][:][:, 0], A[indices][0][:][:, 1], marker='o', s=250, color='k', facecolors='none')
# plt.scatter(test_data[0], test_data[1], marker='x', s=100, color='k')
# plt.show()


# K-Nearest Neighbors Classifier
from sklearn.datasets import *
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

def Image_display(i): # Tạo hàm  để hiển thị figure của 1 số bất kỳ từ tập data
   plt.imshow(digit['images'][i],cmap = 'Greys_r')
   plt.show()
   
digit = load_digits() # Load tập data 
digit_d = pd.DataFrame(digit['data'][0:1600]) # Tạo DataFrame chứa data của 1600 số từ tập data
Image_display(9)

train_x = digit['data'][:1600] # Gán input data của 1600 số vào biến 
train_y = digit['target'][:1600] # Gán input label của 1600 vào biến 
KNN = KNeighborsClassifier(20) # Tạo bộ phân loại KNN với số lượng closest neighbors là 20
KNN.fit(train_x,train_y) # Train

KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski', metric_params = None, n_jobs = 1, n_neighbors = 20, p = 2, weights = 'uniform')

test = np.array(digit['data'][1725]) # Tạo data test với số 1725 ngoài 1600 số
test1 = test.reshape(1,-1) # Reshape test thành mảng 2D
Image_display(1725)

print(KNN.predict(test1)) # Dự đoán label của test1 bằng KNN

print(digit['target_names']) # In tên các label, từ 0-9 trong tập data digit