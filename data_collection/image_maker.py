'''
Handles the functionality of turning characters into pictures.

***python 2 only*** until PIL and pygame release python 3 compatible code,
this must be run in python 2. Thankfully however, this produces files that
can of course be read anywhere.
'''
import os
import pygame
import codecs
import StringIO
from char_util import get_stroke_count
from PIL import Image
from skimage.io import imread, imsave
from skimage.color import rgb2gray

'''
Font file dictionary.

dictionary of all fonts that can be used with the characters.
The key is the name of the font file. The value is a tuple 
corresponding to the fonts offset at size 64.
'''
fonts = { "fangsongti.ttf"    : ( 0,   0) }
          #"hanyisentiwen.ttf" : (-6, -35) ,
          #"simsun.ttc"        : ( 0,   0) ,
          #"ximingti.ttc"      : ( 0,   0) ,
          #"xingshuti.ttf"     : ( 0,   0) ,
          #"xinsongti.ttc"     : ( 0,   0) }


def process_file(filename):
    '''
    Reads in the given file and makes pictures of all the characters present.

    Parameters
    ----------
    filename : string
        a string containing name of the file to be read.
    '''
    f = codecs.open(filename, encoding='utf-8')
    for line in f:
        make_char(line, 256)

def make_char(chars, size):
    '''
    converts a string of chinese characters into pictures. Makes
    one picture for each installed font.

    Parameters
    ----------
    chars : string
        A string containing all the characters to be picturized
    size : int
        The point size of the font (and one side length of the final picture).
    '''
    for font, offset in fonts.items():
        map(lambda c: 
                char_to_pic(c, font ,size 
                        , tuple(size*x/64 for x in offset)) 
                , chars)

def char_to_pic(character, fontFile, size = 64, offset = (0,0)):
    '''
    Uses a given font file to convert a given character into an image.

    Parameters
    ----------
    character : unicode character
        The character of the final image.

    fontFile : string
        the file to get the font data from.

    size : int
        the size of the image.

    offset : (int, int)
        the display offset of the character. Only needed for some
        naughty fonts.
    '''

    pygame.init()
    font = pygame.font.Font(os.path.join("./fonts", fontFile), size)
    text = character #.decode('utf-8')
    imgName = "../data/" \
        + str(get_stroke_count(text)) + "." \
        + text + "." \
        + os.path.splitext(fontFile)[0] \
        + ".png"
    render(text,font,imgName, (size,size), offset)



def render(text, font, imgName, dims, offset = (0, 0)):
    '''
    Takes preprocessed parts and converts them into an image.

    Parameters
    ----------
    text : unicode string
        the text to be put on the image.
    font : string
        the pygame font handle to be used to display the character.
    dims : int
        the dimensions of the final picture.
    offset : (int, int)
        the offset of the character in the picture.
    '''
    
    im = Image.new("L", dims, 255)
    rtext = font.render(text, True, (0, 0, 0), (255, 255, 255))
    sio = StringIO.StringIO()
    pygame.image.save(rtext, sio)
    sio.seek(0)
    line = Image.open(sio)
    im.paste(line, offset)
    im.save(imgName)
