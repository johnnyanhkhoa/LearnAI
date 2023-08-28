import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_data(input_file):
    input_data = np.loadtxt(input_file)
    
    dates = pd.date_range('1950-01', periods=input_data.shape[0], freq='M')
    
    df = pd.DataFrame(input_data, index=dates, columns=['Column1', 'Column2', 'Column3'])
    return df

if __name__ == '__main__':
    input_file = r'E:\Projects\LearnAI\Input\AO.txt'
    
    # Đọc dữ liệu và vẽ biểu đồ cho Column 1
    df = read_data(input_file)
    plt.figure()
    df['Column1'].plot()
    plt.title('Column 1')
    plt.show()
    
    # Đọc dữ liệu và vẽ biểu đồ cho Column 2
    plt.figure()
    df['Column2'].plot()
    plt.title('Column 2')
    plt.show()
    
    # Đọc dữ liệu và vẽ biểu đồ cho Column 3
    plt.figure()
    df['Column3'].plot()
    plt.title('Column 3')
    plt.show()
    
    # Vẽ biểu đồ cho Column 3 trong khoảng thời gian 1980-1990
    plt.figure()
    df['Column3']['1980':'1990'].plot()
    plt.title('Column 3 (1980-1990)')
    plt.show()
    
    # Tính trung bình của Column 3
    column3_mean = df['Column3'].mean()
    print('Trung bình của Column 3:', column3_mean)
