import keras
import os
from PIL import Image
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.utils import to_categorical
 
# 全局变量
batch_size = 4
nb_classes = 3
epochs = 5
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)



def load_data(file_name):
    file_list = os.listdir(file_name)
    img_list = []
    for i in file_list:        
        img = Image.open(file_name+'/'+i)
        img = img.resize((200, 200),Image.ANTIALIAS)
        # print(np.array(img).shape)
        # >>> y = np.expand_dims(x, axis=0)

        img_list.append(np.expand_dims(np.array(img)/255, axis=2))
    return np.array(img_list)


def make_label():
    y = [0 for i in range(0,201)]+[1 for i in range(0,201)]+[2 for i in range(0,201)]
    return to_categorical(y)
    

def train(x,y):


    model = Sequential()
    """
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='same',
                        input_shape=input_shape))
    """
    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='same',
                        input_shape=(200,200,1))) # 卷积层1
    model.add(Activation('relu')) #激活层
    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]))) #卷积层2
    model.add(Activation('relu')) #激活层
    model.add(MaxPooling2D(pool_size=pool_size)) #池化层
    model.add(Dropout(0.25)) #神经元随机失活
    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))# 卷积层1
    model.add(Activation('relu')) #激活层
    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]))) #卷积层2
    model.add(Activation('relu')) #激活层
    model.add(MaxPooling2D(pool_size=pool_size)) #池化层
    model.add(Flatten()) #拉成一维数据
    model.add(Dense(64)) #全连接层1
    model.add(Activation('relu')) #激活层
    model.add(Dropout(0.5)) #随机失活
    model.add(Dense(nb_classes)) #全连接层2
    model.add(Activation('softmax')) #Softmax评分
 
    #编译模型
    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
    #训练模型
    model.fit(x, y, batch_size=batch_size, epochs=epochs)
    model.save('my_moedel.h5')




if __name__ == '__main__':
    x1 = load_data('./data/hujian')
    # print(x1.shape)
    x2 = load_data('./data/tangqi')
    x3 = load_data('./data/Yehaoze')
    x = np.vstack((x1,x2))
    x = np.vstack((x,x3))
    y = make_label()
    # print(x.shape,y[0])
    train(x,y)