#from main3 import TCSOCR #use for 64 X 512 images
from main import TCSOCR #use for 64 X 128 images
import datetime

ocr = TCSOCR()
run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
ocr.train(run_name,0,2,128)
#ocr.train(name_of_the_folder, starting epoch , ending epoch, width of image)
