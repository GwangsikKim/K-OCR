import numpy as np
import json
import matplotlib.pylab as plt
from PIL import Image
from keras import models, layers

def test_Image(addr,table):
    T_Image = Image.open(addr)
    T_Image = T_Image.resize((32,32))
    T_Image_Array = np.array(T_Image,'uint8')
    plt.imshow(T_Image_Array)
    T_Image_Array = T_Image_Array.reshape(1,32,32,3)
    a = CNN.predict(T_Image_Array)
    b = np.argmax(a,axis=1)
    print('예측한 음절: ',table[str(b[0])])

CNN = models.load_model('model/Korean_CNN_model.h5')
with open('model/index_to_syllable.json','r',encoding='utf-8') as f:
    index_to_syllable = json.load(f)
CNN.summary()

T_img_file = "training/data/3_syllable/01107150.png"

test_Image(T_img_file,index_to_syllable)
