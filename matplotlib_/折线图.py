import matplotlib.pylab as plt
import  numpy as np
plt.rcParams['font.sans-serif']="SimHei"#设置默认字体为黑体
plt.rcParams["axes.unicode_minus"]=False#正常显示负号
n=24
#生成随机数
y1=np.random.randint(27,37,n)
y2=np.random.randint(40,60,n)
#绘制折线图和设置右上角图例
plt.plot(y1,label="温度")
plt.plot(y2,label="湿度")

plt.xlim(0,23)
plt.ylim(20,70)

plt.xlabel("时间",fontsize=12)
plt.ylabel("测量值",fontsize=12)
#显示图例
plt.legend()
plt.show()
