'''Logic Programming''' 
from kanren import run, var, fact
from kanren.assoccomm import eq_assoccomm as eq
from kanren.assoccomm import commutative, associative


add = 'add' # Tạo biến đại diện cho các toán tử phép cộng
mul = 'mul' # Tạo biến đại diện cho các toán tử phép nhân 

fact(commutative, mul) # Khai báo quy tắc đến tính giao hoán trong phép nhân
fact(commutative, add) # Khai báo quy tắc đến tính giao hoán trong phép cộng
fact(associative, mul)  # Khai báo quy tắc đến tính kết hợp trong phép nhân
fact(associative, add)  # Khai báo quy tắc đến tính kết hợp trong phép cộng

a, b = var('a'), var('b')

original_pattern = (mul, (add, 5, a), b) # Tạo biểu thức logic dùng để so sánh với các biểu thức khác. Ở đây là phép nhân của phép cộng giữa 5 và a, kết quả được nhân với b

exp1 = (mul, (add, 5, 3), 2) # Tạo biểu thức để so sáng với original_pattern
exp2 = (add,5,(mul,8,1))

print(run(0, (a,b), eq(original_pattern, exp1))) # Áp dụng hàm run() để tìm các giá trị của a và b trong original_pattern khớp với exp1
print(run(0, (a,b), eq(original_pattern, exp2))) # Áp dụng hàm run() để tìm các giá trị của a và b trong original_pattern khớp với exp2


'''Kiểm tra số nguyên tố'''
from kanren import isvar, run, membero
from kanren.core import success, fail, goaleval, condeseq, eq, var
from sympy.ntheory.generate import prime, isprime # Cung cấp các hàm liên quan đến số nguyên tố
import itertools as it

def prime_check(x): # Tạo hàm logic để kiểm tra số nguyên tố
    if isvar(x):
        return condeseq([(eq,x,p)] for p in map(prime, it.count(1))) # Trả về logic tương ứng kiểm tra xem x có phải số nguyên tố hay không. Nếu x là var -> được sử dụng trong quá trình giải quyết ràng buộc logic để tìm giá trị x thỏa mãn.
    else:
        return success if isprime(x) else fail # Trả success nếu là số nguyên tố và ngược lại là fail
    
x = var() # Tạo logic var
print((set(run(0,x,(membero,x,(12,14,15,19,20,21,22,23,29,30,41,44,52,62,65,85)),(prime_check,x))))) # Tìm các giá trị của x trong set mà thỏa mãn cả 2 điều kiện: x thuộc set và x là số nguyên tố. Result là 1 set chứa các x value thỏa mãn
print((run(10,x,prime_check(x)))) # Tìm 10 giá trị đầu tiên của x thỏa mãn prime_check(x)