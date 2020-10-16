# Python有五个标准的数据类型：
#
# Numbers（数字）
# String（字符串）
# List（列表）
# Tuple（元组）
# Dictionary（字典）



# Python支持四种不同的数字类型：
#
# int（有符号整型）
# long（长整型[也可以代表八进制和十六进制]）
# float（浮点型）
# complex（复数）
import  sys

var1=10
var2=10
del  var2
print(var1)

s="abcdef"
s1=s[0:6]
sys.stdout.write(s1+"\n")

List=['root',1,10.3,'jion']
List[2]=3 # 复制操作
TinyList=['buttom','john',2]
print(List*2) # 输出列表两次
print(List+TinyList)# 打印组合的列表

# 元组用 () 标识。内部元素用逗号隔开。但是元组不能二次赋值，相当于只读列表。
tuple=(1,2,3,4,5,6)
print ("tuple第二是",tuple[1])

dict = {}
dict['one'] = "This is one"
dict[2] = "This is two"

tinydict = {'name': 'runoob', 'code': 6734, 'dept': 'sales'}

