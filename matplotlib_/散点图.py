import matplotlib.pylab as plt
import  numpy as np
plt.rcParams['font.sans-serif']="SimHei"#设置默认字体为黑体
plt.rcParams["axes.unicode_minus"]=False#正常显示负号
n=1024
#标准正太分布
x=np.random.normal(0,1,n)#生成正太分布x
y=np.random.normal(0,1,n)
#绘制散点图
plt.scatter(x,y,color="blue",marker='*')#形状为*
#设置标题
plt.title("标准正态分布",fontsize=20)
#设置文字标签
plt.text(2.5,2.5,"均值： 0\n标准差：1")#2.5 2.5表示标签的位置
#设置坐标轴的范围
plt.xlim(-4,4)
plt.ylim(-4,4)
#设置坐标轴的标签
plt.xlabel("横坐标x",fontsize="14")
plt.ylabel("纵坐标y",fontsize="14")
plt.show()