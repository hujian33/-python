import keras
from PIL import Image
import numpy as np


name_dic = {0:'hujian',1:'tangqi',2:'Yehaoze'}

def load_pic(file):
    img = Image.open(file)
    img = img.resize((200, 200),Image.ANTIALIAS)
    img = np.expand_dims(np.array(img)/255,axis=2)
    return np.expand_dims(img,axis=0)

model = keras.models.load_model('./model/my_moedel.h5')
# print(model.summary())
result = model.predict(load_pic('./data/hujian/5.jpg'))
print(np.max(result))
print(name_dic[np.argmax(result)])