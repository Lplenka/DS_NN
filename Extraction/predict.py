from __future__ import unicode_literals
from main3 import TCSOCR
from PIL import Image

ocr = TCSOCR()
ocr.loadWeights('weights05_64x512.h5') # replace with your trained weight file



sample = Image.open('Image64X512/img_9.png')
ans = ocr.ocr_frompic(image = sample)
#ans is a list having only one element
print ('The predicted string is -',''.join(ans))
