'''
Import this file to use the following functions strictly made to help TCSOCR
'''


import os
import numpy as np
from PIL import Image

#converts a 4D array to PIL image and saves it
def array_to_img(b):
    for i in range(15):
        f = b[i : i+1]
        f = np.squeeze(f,axis=0)
        f = np.squeeze(f,axis=2)
        f = f.T
        f = f*255
        image = Image.fromarray(f.astype('uint8'), 'L')
        #return image
        fname = '{prefix}_{index}.{format}'.format(prefix='img',index=i+1,format='png')
        image.save(os.path.join(fname))

#converts a  PIL image to 4D array
def img_to_array(imagepath):
    image = pil_image.open(imagepath)
    a = np.array(image)
    a = a.T
    a = a/255
    a = np.expand_dims(a,0)
    a = np.expand_dims(a,3)
    return a
