import cv2
import sys
import os
from keras.models import load_model
import numpy as np

# 名字标签
name_dic = {0:'HuJian',1:'TangQi',2:'YeHaoZe'}


model = load_model('./model/my_moedel.h5')

color = (0,255,0)

cap = cv2.VideoCapture(1)

# 人脸识别分类器本地存储路径
cascade_path = "D:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"



while True:
    ret, frame = cap.read()
    if ret is True:
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        continue
    cascade = cv2.CascadeClassifier(cascade_path)
    faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=2, minSize=(32, 32))
    if len(faceRects) > 0:
        for faceRect in faceRects:
            x, y, w, h = faceRect
            image = frame_gray[y: y + h, x: x + w]
            pic = cv2.resize(image, (200, 200), interpolation=cv2.INTER_CUBIC)
            img = np.expand_dims(np.array(pic)/255,axis=2)
            result = model.predict(np.expand_dims(img, axis=0))
            name = ''
            if np.max(result) > 0.8:
                name = name_dic[np.argmax(result)]
            else:
                name = 'unknown'
            print(name)
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
            cv2.putText(frame,name,(x + 30, y + 30),  cv2.FONT_HERSHEY_SIMPLEX,  1,  (255, 0, 255),  2)
    cv2.imshow('face_recognition',frame)
    k = cv2.waitKey(10)
        # 如果输入q则退出循环
    if k & 0xFF == ord('q'):
        break

# 释放摄像头并销毁所有窗口
cap.release()
cv2.destroyAllWindows()
