'''
This main file is a convolutional stack followed by a recurrent stack
and a CTC logloss function to perform optical character recognition
of generated text images.
This starts off with 4 letter words.  For the first 12 epochs, the
difficulty is gradually increased using the TextImageGenerator class
which is both a generator class for test/train data and a Keras
callback class. After 20 epochs, longer sequences are thrown at it
by recompiling the model to handle a wider image and rebuilding
the word list to include two words separated by a space.
The table below shows normalized edit distance values. Theano uses
a slightly different CTC implementation, hence the different results.

This file can be used to process 64x512 image.
The model is trained with lowercase & space only.
Parameters can be changed to use it for different size of images and
training can be expanded for more charaters.

'''
import os
import itertools
import re
import datetime
import cairocffi as cairo
import editdistance
import numpy as np
from scipy import ndimage
import pylab
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD , Adam
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
import simplejson
from keras.models import model_from_json
from PIL import ImageOps

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None


OUTPUT_DIR = 'img_ocr'

np.random.seed(55)
counter = 0
# this creates larger "blotches" of noise which look
# more realistic than just adding gaussian noise
# assumes greyscale with pixels ranging from 0 to 1

from PIL import Image, ImageDraw, ImageFont
from PIL import PngImagePlugin
from random import choice as oldchoice
from random import randint as randint






def sample_background(picture):
    picture = picture.convert('RGB')
    return int(np.median([picture.getpixel((0, 0))[0], \
                          picture.getpixel((0, picture.getbbox()[3] / 2))[0], \
                          picture.getpixel((0, picture.getbbox()[3] - 1))[0], \
 \
                          picture.getpixel((picture.getbbox()[2] / 2, 0))[0], \
                          picture.getpixel((picture.getbbox()[2] / 2, picture.getbbox()[3] - 1))[0], \
 \
                          picture.getpixel((picture.getbbox()[2] - 1, 0))[0], \
                          picture.getpixel((picture.getbbox()[2] - 1, picture.getbbox()[3] / 2))[0], \
                          picture.getpixel((picture.getbbox()[2] - 1, picture.getbbox()[3] - 1))[0]
                          ]))

def size_randpad(picture):
    """Padding a PIL image to a random position and returning it as a keras friendly matrix"""
    backc = sample_background(picture)
    newpic = Image.new('RGB', (512, 64), (backc, backc, backc, 255))
    orw = picture.getbbox()[2]
    orh = picture.getbbox()[3]
    new = newpic.getbbox()[2]
    neh = newpic.getbbox()[3]

    if orw > new or orh > neh:
        wr = float(new) / float(orw)
        hr = float(neh) / float(orh)
        rat = min([wr, hr]) * .9
        picture = picture.resize((int(orw * rat) - 1, int(orh * rat) - 1), Image.LANCZOS)
        orw = picture.getbbox()[2]
        orh = picture.getbbox()[3]

    if orh > float(neh) * .5:
        multw = float(randint(int(orw * .5), new)) / float(orw)
        multh = float(randint(int(orh * .5), neh)) / float(orh)
    else:
        multw = float(randint(orw, new)) / float(orw)
        multh = float(randint(orh, neh)) / float(orh)
    mult = min([multw, multh])
    orw = int(float(orw) * float(mult))
    orh = int(float(orh) * float(mult))
    picture = picture.resize((orw, orh), Image.LANCZOS)
    randoffw = randint(0, new - orw)
    randoffh = randint(0, neh - orh)
    tup = (randoffw, randoffh)
    newpic.paste(picture, tup)
    newpic = newpic.convert("L")
    numparr = np.array(newpic)
    numparr = numparr.astype(np.float32) / 255
    numparr = numparr.transpose()[None, :, :]
    return numparr



def array_to_img(x, data_format=None, scale=True):
    """Converts a 3D Numpy array to a PIL Image instance.
    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
        scale: Whether to rescale image values
            to be within [0, 255].
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=K.floatx())
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)
    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])



def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())
    # x = x.transpose()
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


def load_img(path, grayscale=False, target_size=None):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size:
        hw_tuple = (target_size[1], target_size[0])
        if img.size != hw_tuple:
            img = img.resize(hw_tuple)
    return img


def speckle(img):
    severity = np.random.uniform(0, 0.6)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    return img_speck


# paints the string in a random location the bounding box
# also uses a random font, a slight random rotation,
# and a random amount of speckle noise

def paint_text(text, w, h, rotate=False, ud=False, multi_fonts=False):
    global counter
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    with cairo.Context(surface) as context:
        context.set_source_rgb(1, 1, 1)  # White
        context.paint()
        # this font list works in Centos 7
        if multi_fonts:
            fonts = ['Century Schoolbook', 'Courier', 'STIX', 'URW Chancery L', 'FreeMono']
            context.select_font_face(np.random.choice(fonts), cairo.FONT_SLANT_NORMAL,
                                     np.random.choice([cairo.FONT_WEIGHT_BOLD, cairo.FONT_WEIGHT_NORMAL]))
        else:
            context.select_font_face('Courier', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(25)
        box = context.text_extents(text)
        border_w_h = (4, 4)
        if box[2] > (w - 2 * border_w_h[1]) or box[3] > (h - 2 * border_w_h[0]):
            raise IOError('Could not fit string into image. Max char count is too large for given image width.')

        # teach the RNN translational invariance by
        # fitting text box randomly on canvas, with some room to rotate
        max_shift_x = w - box[2] - border_w_h[0]
        max_shift_y = h - box[3] - border_w_h[1]
        top_left_x = np.random.randint(0, int(max_shift_x))
        if ud:
            top_left_y = np.random.randint(0, int(max_shift_y))
        else:
            top_left_y = h // 2
        context.move_to(top_left_x - int(box[0]), top_left_y - int(box[1]))
        context.set_source_rgb(0, 0, 0)
        context.show_text(text)

    buf = surface.get_data()
    a = np.frombuffer(buf, np.uint8)
    a.shape = (h, w, 4)
    a = a[:, :, 0]  # grab single channel
    a = a.astype(np.float32) / 255
    a = np.expand_dims(a, 0)
    if rotate:
        a = image.random_rotation(a, 3 * (w - top_left_x) / w + 1)
    a = speckle(a)
    #To save the test/train images
    # counter += 1
    # img = array_to_img(a, 'channels_first', scale=True)
    # fname = '{prefix}_{index}.{format}'.format(prefix='img',index=counter,format='png')
    # img.save(os.path.join('data', fname))
    return a


def shuffle_mats_or_lists(matrix_list, stop_ind=None):
    ret = []
    assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
    len_val = len(matrix_list[0])
    if stop_ind is None:
        stop_ind = len_val
    assert stop_ind <= len_val

    a = list(range(stop_ind))
    np.random.shuffle(a)
    a += list(range(stop_ind, len_val))
    for mat in matrix_list:
        if isinstance(mat, np.ndarray):
            ret.append(mat[a])
        elif isinstance(mat, list):
            ret.append([mat[i] for i in a])
        else:
            raise TypeError('`shuffle_mats_or_lists` only supports '
                            'numpy.array and list objects.')
    return ret


def text_to_labels(text, num_classes):
    ret = []
    for char in text:
        if char >= 'a' and char <= 'z':
            ret.append(ord(char) - ord('a'))
        elif char == ' ':
            ret.append(26)
        elif char >= 'A' and char <= 'Z':
            ret.append( 27 + ord(char) - ord('A'))
        elif char >= '0' and char <= '9':
            ret.append( 53 + ord(char) - ord('0'))
        elif char == '.':
            ret.append(63)
        elif char == '/':
            ret.append(64)
        elif char == '-':
            ret.append(65)
    return ret


# only a-z and space..probably not to difficult
# to expand to uppercase and symbols

def is_valid_str(in_str):
    return True


# Uses generator functions to supply train/test with
# data. Image renderings are text are created on the fly
# each time with random perturbations

class TextImageGenerator(keras.callbacks.Callback):

    def __init__(self, monogram_file, bigram_file, minibatch_size,
                 img_w, img_h, downsample_factor, val_split,
                 absolute_max_string_len=16):

        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.monogram_file = monogram_file
        self.bigram_file = bigram_file
        self.downsample_factor = downsample_factor
        self.val_split = val_split
        self.blank_label = self.get_output_size() - 1
        self.absolute_max_string_len = absolute_max_string_len

    def get_output_size(self):
        return 67

    # num_words can be independent of the epoch size due to the use of generators
    # as max_string_len grows, num_words can grow
    def build_word_list(self, num_words, max_string_len=None, mono_fraction=0.5):
        assert max_string_len <= self.absolute_max_string_len
        assert num_words % self.minibatch_size == 0
        assert (self.val_split * num_words) % self.minibatch_size == 0
        self.num_words = num_words
        self.string_list = [''] * self.num_words
        tmp_string_list = []
        self.max_string_len = max_string_len
        self.Y_data = np.ones([self.num_words, self.absolute_max_string_len]) * -1
        self.X_text = []
        self.Y_len = [0] * self.num_words

        # monogram file is sorted by frequency in english speech
        with open(self.monogram_file, 'rt') as f:
            for line in f:
                if len(tmp_string_list) == int(self.num_words * mono_fraction):
                    break
                word = line.rstrip()
                if max_string_len == -1 or max_string_len is None or len(word) <= max_string_len:
                    tmp_string_list.append(word)

        # bigram file contains common word pairings in english speech
        with open(self.bigram_file, 'rt') as f:
            lines = f.readlines()
            for line in lines:
                if len(tmp_string_list) == self.num_words:
                    break
                columns = line.lower().split()
                word = columns[0] + ' ' + columns[1]
                if is_valid_str(word) and \
                        (max_string_len == -1 or max_string_len is None or len(word) <= max_string_len):
                    tmp_string_list.append(word)
        if len(tmp_string_list) != self.num_words:
            raise IOError('Could not pull enough words from supplied monogram and bigram files. ')
        # interlace to mix up the easy and hard words
        self.string_list[::2] = tmp_string_list[:self.num_words // 2]
        self.string_list[1::2] = tmp_string_list[self.num_words // 2:]

        for i, word in enumerate(self.string_list):
            self.Y_len[i] = len(word)
            self.Y_data[i, 0:len(word)] = text_to_labels(word, self.get_output_size())
            self.X_text.append(word)
        self.Y_len = np.expand_dims(np.array(self.Y_len), 1)

        self.cur_val_index = self.val_split
        self.cur_train_index = 0

    # each time an image is requested from train/val/test, a new random
    # painting of the text is performed
    def get_batch(self, index, size, train):
        global counter
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        if K.image_data_format() == 'channels_first':
            X_data = np.ones([size, 1, self.img_w, self.img_h])
        else:
            X_data = np.ones([size, self.img_w, self.img_h, 1])

        labels = np.ones([size, self.absolute_max_string_len])
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        source_str = []
        for i in range(0, size):
            # Mix in some blank inputs.  This seems to be important for
            # achieving translational invariance
            if train and i > size - 4:
                if K.image_data_format() == 'channels_first':
                    X_data[i, 0, 0:self.img_w, :] = self.paint_func('')[0, :, :].T
                else:
                    X_data[i, 0:self.img_w, :, 0] = self.paint_func('',)[0, :, :].T
                labels[i, 0] = self.blank_label
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = 1
                source_str.append('')
            else:
                if K.image_data_format() == 'channels_first':
                    X_data[i, 0, 0:self.img_w, :] = self.paint_func(self.X_text[index + i])[0, :, :].T
                else:
                    X_data[i, 0:self.img_w, :, 0] = self.paint_func(self.X_text[index + i])[0, :, :].T
                labels[i, :] = self.Y_data[index + i]
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = self.Y_len[index + i]
                source_str.append(self.X_text[index + i])
        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        # counter += 1
        # if not os.path.exists('numpydata/'+str(counter)):
        #     os.makedirs('numpydata/'+str(counter))
        # np.save('numpydata/'+str(counter)+'/X_data.npy',X_data)
        # np.save('numpydata/'+str(counter)+'/labels.npy',labels)
        # np.save('numpydata/'+str(counter)+'/input_length.npy',input_length)
        # np.save('numpydata/'+str(counter)+'/label_length.npy',label_length)
        # np.save('numpydata/'+str(counter)+'/source_str.npy',np.array(source_str))
        return (inputs, outputs)

    def next_train(self):
        while 1:
            ret = self.get_batch(self.cur_train_index, self.minibatch_size, train=True)
            self.cur_train_index += self.minibatch_size
            if self.cur_train_index >= self.val_split:
                self.cur_train_index = self.cur_train_index % 32
                (self.X_text, self.Y_data, self.Y_len) = shuffle_mats_or_lists(
                    [self.X_text, self.Y_data, self.Y_len], self.val_split)
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.cur_val_index, self.minibatch_size, train=False)
            self.cur_val_index += self.minibatch_size
            if self.cur_val_index >= self.num_words:
                self.cur_val_index = self.val_split + self.cur_val_index % 32
            yield ret

    def on_train_begin(self, logs={}):
        self.build_word_list(16000, 6, 1)
        self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                  rotate=False, ud=False, multi_fonts=False)

    def on_epoch_begin(self, epoch, logs={}):
        # rebind the paint function to implement curriculum learning
        if epoch >= 3 and epoch < 6:
            self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                      rotate=False, ud=True, multi_fonts=False)
        elif epoch >= 6 and epoch < 9:
            self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                      rotate=False, ud=True, multi_fonts=True)
        elif epoch >= 9:
            self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                      rotate=True, ud=True, multi_fonts=True)
        if epoch >= 21 and self.max_string_len < 12:
            self.build_word_list(32000, 12, 0.5)


# the actual loss calc occurs here despite it not being
# an internal Keras loss function

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        # 26 is space, 27 is CTC blank char
        outstr = ''
        for c in out_best:
            if c >= 0 and c < 26:
                outstr += chr(c + ord('a'))
            elif c == 26:
                outstr += ' '
            elif c >= 27 and c < 53:
                outstr += chr(c + ord('A') - 27)
            elif c >= 53 and c < 63:
                outstr += chr(c + ord('0') - 53)
            elif c == 63:
                outstr += '.'
            elif c == 64:
                outstr += '/'
            elif c == 65:
                outstr += '-'
        ret.append(outstr)
    return ret


class VizCallback(keras.callbacks.Callback):

    def __init__(self, run_name, test_func, text_img_gen, model):
        self.test_func = test_func
        self.output_dir = os.path.join(
            OUTPUT_DIR, run_name)
        self.text_img_gen = text_img_gen
        self.num_display_words = 6
        self.model = model
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:
            word_batch = next(self.text_img_gen)[0]
            num_proc = min(word_batch['the_input'].shape[0], num_left)
            decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
            for j in range(0, num_proc):
                edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print('\nOut of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
              % (num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(os.path.join(self.output_dir, 'weights%02d.h5' % (epoch)))
        # if not os.path.exists('numpydata/'+str(counter)):
        #     os.makedirs('numpydata/'+str(counter))
        model_json = self.model.to_json()
        with open(os.path.join(self.output_dir) + '/model_'+ str(epoch)+'.json', "w") as json_file:
            json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))

        self.show_edit_distance(256)
        word_batch = next(self.text_img_gen)[0]
        res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words])
        if word_batch['the_input'][0].shape[0] < 256:
            cols = 2
        else:
            cols = 1
        for i in range(self.num_display_words):
            pylab.subplot(self.num_display_words // cols, cols, i + 1)
            if K.image_data_format() == 'channels_first':
                the_input = word_batch['the_input'][i, 0, :, :]
            else:
                the_input = word_batch['the_input'][i, :, :, 0]
            pylab.imshow(the_input.T, cmap='Greys_r')
            pylab.xlabel('Truth = \'%s\'\nDecoded = \'%s\'' % (word_batch['source_str'][i], res[i]))
        fig = pylab.gcf()
        fig.set_size_inches(10, 13)
        pylab.savefig(os.path.join(self.output_dir, 'e%02d.png' % (epoch)))
        pylab.close()


# def train(run_name, start_epoch, stop_epoch, img_w):
class TCSOCR:
    def __init__(self, img_w=512, labeltype_hinting=True, verbose=1):
        # Input Parameters
        self.img_h = 64
        self.words_per_epoch = 10
        self.val_split = 0.2
        self.val_words = int(self.words_per_epoch * (self.val_split))

        # Network parameters
        self.conv_filters = 16
        self.kernel_size = (3, 3)
        self.pool_size = 2
        self.time_dense_size = 32
        self.rnn_size = 512

        self.minibatch_size=32


        if K.image_data_format() == 'channels_first':
            input_shape = (1, img_w, self.img_h)
        else:
            input_shape = (img_w, self.img_h, 1)


        self.img_gen = TextImageGenerator(monogram_file=os.path.join(os.getcwd(), 'wordlist.txt'),
                                    bigram_file=os.path.join(os.getcwd(), 'bigram_wordlist.txt'),
                                    minibatch_size= 32,
                                    img_w=img_w,
                                    img_h=self.img_h,
                                    downsample_factor=(self.pool_size ** 2),
                                    val_split=self.words_per_epoch - self.val_words
                                    )
        act = 'relu'
        self.input_data = Input(name='the_input', shape=input_shape, dtype='float32')
        inner = Conv2D(self.conv_filters, self.kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(self.input_data)
        inner = MaxPooling2D(pool_size=(self.pool_size, self.pool_size), name='max1')(inner)
        inner = Conv2D(self.conv_filters, self.kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
        inner = MaxPooling2D(pool_size=(self.pool_size, self.pool_size), name='max2')(inner)

        conv_to_rnn_dims = (img_w // (self.pool_size ** 2), (self.img_h // (self.pool_size ** 2)) * self.conv_filters)
        inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

        # cuts down input size going into RNN:
        inner = Dense(self.time_dense_size, activation=act, name='dense1')(inner)

        # Two layers of bidirecitonal GRUs
        # GRU seems to work as well, if not better than LSTM:
        gru_1 = GRU(self.rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
        gru_1b = GRU(self.rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = GRU(self.rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(self.rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

        # transforms RNN output to character activations:
        self.inner = Dense(self.img_gen.get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
        y_pred = Activation('softmax', name='softmax')(self.inner)

        Model(inputs=self.input_data, outputs=y_pred).summary()

        labels = Input(name='the_labels', shape=[self.img_gen.absolute_max_string_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        # clipnorm seems to speeds up convergence
        #use one of these two
        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        #adam= Adam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6, clipnorm=5)


        self.model = Model(inputs=[self.input_data, labels, input_length, label_length], outputs=loss_out)

        self.test_func = K.function([self.input_data], [y_pred])

    def loadWeights(self, filewithpath):
        if filewithpath:
            self.model.load_weights(filewithpath)

    def train(self, run_name, start_epoch, stop_epoch, img_w):
        words_per_epoch = 10 #should be 16000
        val_split = 0.2
        val_words = int(words_per_epoch * (val_split))

        self.img_gen = TextImageGenerator(monogram_file=os.path.join(os.getcwd(), 'wordlist.txt'),
                                    bigram_file=os.path.join(os.getcwd(), 'bigram_wordlist.txt'),
                                     minibatch_size=32,
                                     img_w=img_w,
                                     img_h=self.img_h,
                                     downsample_factor=(self.pool_size ** 2),
                                     val_split=self.words_per_epoch - self.val_words
                                     )

        adam = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        #adam= Adam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6, clipnorm=5)

        output_dir = os.path.join(OUTPUT_DIR, run_name)


        labels = Input(name='the_labels', shape=[self.img_gen.absolute_max_string_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        y_pred = Activation('softmax', name='softmax')(self.inner)

        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        self.model = Model(inputs=[self.input_data, labels, input_length, label_length], outputs=loss_out)

        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, metrics=['accuracy'], optimizer=adam)

        if start_epoch > 0:
                weight_file = os.path.join(OUTPUT_DIR, os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
                self.model.load_weights(weight_file)


        viz_cb = VizCallback(run_name, self.test_func, self.img_gen.next_val(),self.model)


        self.model.fit_generator(generator=self.img_gen.next_train(), steps_per_epoch=(self.words_per_epoch - self.val_words),
                        epochs=stop_epoch, validation_data=self.img_gen.next_val(), validation_steps=self.val_words,
                        callbacks=[viz_cb, self.img_gen], initial_epoch=start_epoch, verbose=1)


    def ocr_frompic(self, image, debug=False):
        a = np.array(image)
        a = a.T
        a = a/255
        a = np.expand_dims(a,0)
        a = np.expand_dims(a,3)
        res = decode_batch(self.test_func, a[0:1])
        return res
