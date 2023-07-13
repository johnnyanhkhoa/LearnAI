'''Single Variable Regressor'''
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt


input = 'E:\Projects\LearnAI\Input\Mul_linear.txt'

input_data = np.loadtxt(input, delimiter=',') # Load data vào
X, y = input_data[:, :-1], input_data[:, -1] # Chia thành matrix đặc trưng X và var mục tiêu y. Cột cuối của input_data được gán cho y còn lại các cột được gán cho X

training_samples = int(0.6 * len(X)) # Tính số mãu data được train và số data test. Số data train là 60% và còn lại là test
testing_samples = len(X) - training_samples # Số data train là 60% và còn lại là test

X_train, y_train = X[:training_samples], y[:training_samples] # Chia data thành tập train và tập test

X_test, y_test = X[training_samples:], y[training_samples:] # Chia data thành tập train và tập test

reg_linear = linear_model.LinearRegression() # Tạo đối tượng hồi quy tuyến tính 

reg_linear.fit(X_train, y_train) # Train mô hình

y_test_pred = reg_linear.predict(X_test) # Sử dụng X_test để dự đoán giá trị mục tiêu

plt.scatter(X_test, y_test, color = 'red') # Chọn màu cho biểu đồ
plt.plot(X_test, y_test_pred, color = 'black', linewidth = 2) # Vẽ đường cong hồi quy tuyến tính dự đoán với màu 'black' và độ dày 2
plt.xticks(())
plt.yticks(())
plt.show() # Hiển thị đồ thị

print("Performance of Linear regressor:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2)) # Sai số trung bình tuyệt đối
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2)) # Sai số toàn phương trung bình
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2)) # Sai số tuyệt đối trung vị
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred),2)) # Giải thích của phương sai
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2)) # Điểm R2


# '''Multivariable Regressor'''
# import numpy as np
# from sklearn import linear_model
# import sklearn.metrics as sm
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import PolynomialFeatures


# input = 'E:\Projects\LearnAI\Input\Mul_linear.txt'

# input_data = np.loadtxt(input, delimiter=',')
# X, y = input_data[:, :-1], input_data[:, -1]

# training_samples = int(0.6 * len(X))
# testing_samples = len(X) - training_samples

# X_train, y_train = X[:training_samples], y[:training_samples]

# X_test, y_test = X[training_samples:], y[training_samples:]

# reg_linear_mul = linear_model.LinearRegression()

# reg_linear_mul.fit(X_train, y_train)

# y_test_pred = reg_linear_mul.predict(X_test)

# print("Performance of Linear regressor:")
# print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
# print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
# print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
# print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
# print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# polynomial = PolynomialFeatures(degree = 10)
# X_train_transformed = polynomial.fit_transform(X_train)
# datapoint = [[2.23, 1.35, 1.12]]
# poly_datapoint = polynomial.fit_transform(datapoint)

# poly_linear_model = linear_model.LinearRegression()
# poly_linear_model.fit(X_train_transformed, y_train)
# print("\nLinear regression:\n", reg_linear_mul.predict(datapoint)) # Em cần hỗ trợ chỗ này ạ!
# print("\nPolynomial regression:\n", poly_linear_model.predict(poly_datapoint))