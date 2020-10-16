
from aip import AipFace
# from picamera import PiCamera#摄像头的调用方法。
import urllib.request
# import RPi.GPIO as GPIO
import base64
import time

# 百度人脸识别API账号信息
APP_ID = '22661812'
API_KEY = 'U3NNWRNRGR7mdX8i37wLg6aC'
SECRET_KEY = 'm4lcctFCugcyCppfVn8jHKvzgRGvFPd1'
client = AipFace(APP_ID, API_KEY, SECRET_KEY)  # 创建一个客户端用以访问百度云
# 图像编码方式
IMAGE_TYPE = 'BASE64'
# camera = PiCamera()#定义一个摄像头对象
# 用户组
GROUP = '445'


# 对图片的格式进行转换
def transimage():
    f = open('home/me/me.jpg', 'rb')  # 打开人脸照片，此处可以更换成摄像头，进行实时的人脸捕捉
    img = base64.b64encode(f.read())
    return img


# 上传到百度api进行人脸检测
def go_api(image):
    result = client.search(str(image, 'utf-8'), IMAGE_TYPE, GROUP);# 在百度云人脸库中寻找有没有匹配的人脸

    if result['error_msg'] == 'SUCCESS':  # 如果成功了
        name = result['result']['user_list'][0]['user_id']  # 获取名字
        score = result['result']['user_list'][0]['score']  # 获取相似度
        if score > 80:  # 如果相似度大于80
            print("欢迎%s !" % name)
            time.sleep(3)
        else:
            print("对不起，我不认识你！")
            name = 'Unknow'
            return 0
        curren_time = time.asctime(time.localtime(time.time()))  # 获取当前时间

        # 将人员出入的记录保存到Log.txt中
        f = open('Log.c', 'wb')
        f.close()
        return 1
    if result['error_msg'] == 'pic not has face':
        print('检测不到人脸')
        time.sleep(2)
        return 0
    else:
        print(result['error_code'] , result['error_code'])
        return 0


# 主函数
if __name__ == '__main__':
    while True:
        print('准备')
        if True:
            # getimage()#拍照
            img = transimage()  # 转换照片格式
            res = go_api(img)  # 将转换了格式的图片上传到百度云
            if (res == 1):  # 是人脸库中的人
                print("开门")  # 可以拓展成开关门的应用
            else:
                print("关门")
                print('稍等三秒进入下一个')
                time.sleep(3)