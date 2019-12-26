###################################################
#收集组员人脸
###################################################
from rlsb.gain_face import  CatchPICFromVideo

while True:
    print("是否录入组员信息(Yes or No)?")
    if input() == 'Yes':
        #组员姓名(要输入英文，汉字容易报错)
        new_user_name = input("请输入您的姓名：")

        print("请看摄像头！")

        #采集员工图像的数量自己设定，越多识别准确度越高，但训练速度贼慢
        window_name = 'data acquisition'                                                                                  #图像窗口
        camera_id = 0                                                                                            #相机的ID号
        images_num = 200                                                                                         #采集图片数量
        path = './data/' + new_user_name   #图像保存位置

        CatchPICFromVideo(window_name,camera_id,images_num,path)
    else:
        break
