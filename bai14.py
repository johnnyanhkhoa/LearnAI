# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# def read_data(input_file):
#     input_data = np.loadtxt(input_file)
    
#     dates = pd.date_range('1950-01', periods=input_data.shape[0], freq='M')
    
#     df = pd.DataFrame(input_data, index=dates, columns=['Column1', 'Column2', 'Column3'])
#     return df

# if __name__ == '__main__':
#     input_file = r'E:\Projects\LearnAI\Input\AO.txt'
    
#     # Đọc dữ liệu và vẽ biểu đồ cho Column 1
#     df = read_data(input_file)
    
#     '''Trích xuất thống kê'''
#     # Tính trung bình, giá trị lớn nhất và giá trị nhỏ nhất của Column 3
#     column3_mean = df['Column3'].mean()
#     column3_max = df['Column3'].max()
#     column3_min = df['Column3'].min()
#     print('Trung bình của Column 3:', column3_mean)
#     print('Giá trị lớn nhất của Column 3:', column3_max)
#     print('Giá trị nhỏ nhất của Column 3:', column3_min)

#     # Hiển thị thông tin tóm tắt về dữ liệu
#     print(df['Column3'].describe())
    
#     '''Re-sampling với mean()'''
#     # Re-sampling dữ liệu theo năm và vẽ biểu đồ
#     timeseries_mm = df['Column3'].resample("A").mean()
#     plt.figure()
#     timeseries_mm.plot(style='g--')
#     plt.title('Re-sampling theo năm và giá trị trung bình')
#     plt.show()
    
#     '''Re-sampling với median()'''
#     # Re-sampling dữ liệu theo năm với median và vẽ biểu đồ
#     timeseries_mm = df['Column3'].resample("A").median()
#     plt.figure()
#     timeseries_mm.plot()
#     plt.title('Re-sampling theo năm và median')
#     plt.show()
    
#     '''Rolling Mean'''
#     # Trung bình trượt (rolling mean) và vẽ biểu đồ
#     rolling_mean = df['Column3'].rolling(window=12, center=False).mean()
#     plt.figure()
#     rolling_mean.plot(style='-g')
#     plt.title('Trung bình trượt (Rolling Mean)')
#     plt.show()
    


'''Phân tích dữ liệu tuần tự bằng mô hình Markov ẩn (HMM)'''
import numpy as np
import datetime
import warnings
import yfinance as yf
from matplotlib import pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates

from hmmlearn.hmm import GaussianHMM

start_date = datetime.date(1995, 10, 10)
end_date = datetime.date(2015, 4, 25)
quotes = yf.download('INTC', start=start_date, end=end_date)

closing_quotes = quotes['Close'].values

volumes = quotes['Volume'].values[1:]

diff_percentages = 100.0 * np.diff(closing_quotes) / closing_quotes[:-1]
dates = mdates.date2num(quotes.index[1:])
training_data = np.column_stack([diff_percentages, volumes])

hmm = GaussianHMM(n_components=7, covariance_type='diag', n_iter=1000)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    hmm.fit(training_data)

num_samples = 300
samples, _ = hmm.sample(num_samples)

plt.figure()
plt.title('Difference percentages')
plt.plot(np.arange(num_samples), samples[:, 0], c='black')

plt.figure()
plt.title('Volume of shares')
plt.plot(np.arange(num_samples), samples[:, 1], c='black')
plt.ylim(ymin=0)
plt.show()

