'''Thuật toán Clustering'''

# # K-Means
# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs

# X, y_true = make_blobs(n_samples = 500, centers = 4, cluster_std = 0.40, random_state = 0) # Sử dụng make_blob để tạo tập data 2 chiều, chứa 4 đốm màu
# plt.scatter(X[:, 0], X[:, 1], s = 50) # Chọn màu cho biểu đồ
# plt.show() # Hiển thị biểu đồ

# kmeans = KMeans(n_clusters = 4) # Khởi tạo kmeans với số lượng cluster = 4
# kmeans.fit(X) # Train mô hình
# y_kmeans = kmeans.predict(X) # Sử dụng X để dự đoán label 
# plt.scatter(X[:, 0], X[:, 1], c = y_kmeans, s = 50, cmap = 'viridis')
# centers = kmeans.cluster_centers_ # Trích xuất cluster center từ mô hình kmeans đã train
# plt.scatter(centers[:, 0], centers[:, 1], c = 'black', s = 200, alpha = 0.5)
# plt.show()

# Mean Shift
import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.datasets import make_blobs

centers = [[2,2],[4,5],[3,10]]
X, _ = make_blobs(n_samples = 500, centers = centers, cluster_std = 1) # Tạo tập data 2d, chứa 4 đốm màu bằng make_blod
plt.scatter(X[:,0],X[:,1])
plt.show()

ms = MeanShift() # Tạo mô hình
ms.fit(X) # Train mô hình 
labels = ms.labels_
cluster_centers = ms.cluster_centers_ # Save lebel của từng điểm data và save tọa độ trung tâm

print(cluster_centers) # Print tọa độ trung tâm
n_clusters_ = len(np.unique(labels)) # Tính số lượng cụm ước lượng
print("Estimated clusters:", n_clusters_)

colors = 10*['r.','g.','b.','c.','k.','y.','m.'] # Tạo list mẫu màu
for i in range(len(X)): # Dùng lặp for để vẽ từng điểm data trên biểu đồ, màu của từng điểm xác định bởi colors[labels[i]]
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)
plt.scatter(cluster_centers[:,0],cluster_centers[:,1],marker = "x",color = 'k', s = 150, linewidths = 5, zorder = 10)
plt.show()