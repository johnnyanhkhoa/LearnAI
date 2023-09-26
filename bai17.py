# '''Constraint Satisfaction Problems (CSPs)'''

# '''Quan hệ đại số'''
# from constraint import *

# problem = Problem() # Tạo đối tượng Problem

# problem.addVariable('a', range(10)) # Lấy biến a nhận value 0-9
# problem.addVariable('b', range(10)) # Lấy biến b nhận value 0-9

# problem.addConstraint(lambda a, b: a * 2 == b) # Thêm ràng buộc bằng hàm lambda buộc b phải double value a

# solutions = problem.getSolutions() # Tìm tất cả các giải pháp thỏa mãn bài toán và yêu cầu. Result cho ra list các bộ value của a và b

# print(solutions)


'''Magic Square'''
def magic_square(matrix_ms):
    iSize = len(matrix_ms[0]) # Tính độ dài của hàng row tiên hoặc số col của matrix_ms để lưu vào iSize
    sum_list = []

    for col in range(iSize): # Lặp qua từng vòng của matrix
        sum_list.append(sum(row[col] for row in matrix_ms)) # Tính tổng các phần tử trong col và append vào sum_list
    
    sum_list.extend([sum(row) for row in matrix_ms]) # Tính tổng các row của matrix = list comprehension và extend sum_list bằng sum này

    dlResult = 0 # Tạo biến để tính sum đường chéo trái của matrix
    for i in range(0, iSize): # Lặp qua từng phần tử trên đường chéo trái
        dlResult += matrix_ms[i][i] # Tính tổng các phần tử bằng cách thêm value của phần tử tại vị trí (i,i) vào dlResult
    sum_list.append(dlResult)
    
    drResult = 0 # Tạo biến để tính sum đường chéo phải của matrix
    for i in range(iSize - 1, -1, -1): # Lặp qua từng phần tử trên đường chéo phải
        drResult += matrix_ms[i][i] # Tính tổng các phần tử bằng cách thêm value của phần tử tại vị trí (i,i) vào drResult
    sum_list.append(drResult)

    target_sum = sum_list[0] # Lấy sum đầu tiên từ list để so sánh
    if all(target_sum == s for s in sum_list): # Kiểm tra xem all sum trong sum_list có == hay !=. Nếu == thì return True
        return True
    return False # Nếu không bằng thì return False

print(magic_square([[1, 2, 3], [4, 5, 6], [7, 8, 9]])) # Không phải ma trận ma thuật
print(magic_square([[3, 9, 2], [3, 5, 7], [9, 1, 6]])) # Không phải ma trận ma thuật



