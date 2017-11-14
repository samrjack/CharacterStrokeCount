# -*- coding: utf-8 -*-

import os
import pygame
import codecs
from PIL import Image
import StringIO

'''
dictionary of all fonts that can be used with the characters.
The key is the name of the font file. The value is a tuple 
corresponding to the fonts offset at size 64.
'''
fonts = { "fangsongti.ttf"    : ( 0,   0) ,
          "hanyisentiwen.ttf" : (-6, -35) ,
          "simsun.ttc"        : ( 0,   0) ,
          "ximingti.ttc"      : ( 0,   0) ,
          "xingshuti.ttf"     : ( 0,   0) ,
          "xinsongti.ttc"     : ( 0,   0) }

'''
Read in the stroke counts of all chinese unicode characters.
'''
strokeCount = {}
with open("count/totalstrokes.txt") as f:
    for line in f:
        (key, val) = line.split()
        strokeCount[key] = val

'''
process_file
    Reads in the given file and makes pictures of all the characters present.

    filename - a string containing name of the file to be read.
'''
def process_file(filename):
    f = codecs.open(filename, encoding='utf-8')
    for line in f:
        make_char(line, 256)

'''
get_stroke_count
    finds the official stroke count of a given unicode chinese character.

    char - a single character to be converted to a stroke count.

    returns - The stroke count if one is found. 0 if no the character isn't
        recognized.
'''
def get_stroke_count(char):
    # Gets the hex number withou 0x at beginning
    hexNum = hex(ord(char))[2:].upper()
    return int(strokeCount.get(hexNum, '0'))

'''
make_char
    converts a string of chinese characters into pictures. Makes
    one picture for each installed font.

    chars - the string to be picturized.
    size - the point size of the font (and one side length of the final picture)
'''
def make_char(chars, size):
    for font, offset in fonts.items():
        map(lambda c: 
                char_to_pic(c, font ,size 
                    , tuple(size*x/64 for x in offset)) 
                , chars)

'''
char_to_pic
    Uses a given font file to convert a given character into an image.

    character - The character of the final image.
    fontFile - the file to get the font data from.
    size - the size of the image.
    offset - the display offset of the character. Only needed for some
        naughty fonts.
'''
def char_to_pic(character, fontFile, size = 64, offset = (0,0)):
    pygame.init()
    font = pygame.font.Font(os.path.join("./fonts", fontFile), size)
    text = character #.decode('utf-8')
    imgName = "./set/" + text + "." \
        + str(size) + "." \
        + os.path.splitext(fontFile)[0] \
        + ".png"
    render(text,font,imgName, (size,size), offset)

'''
render
    Takes preprocessed parts and converts them into an image.

    text - the text to be put on the image.
    font - the pygame font handle to be used to display the character.
    dims - the dimensions of the final picture.
    offset - the offset of the character in the picture.
'''
def render(text, font, imgName, dims, offset = (0, 0)):
    im = Image.new("RGB", dims, (255, 255, 255))
    rtext = font.render(text, True, (0, 0, 0), (255, 255, 255))
    sio = StringIO.StringIO()
    pygame.image.save(rtext, sio)
    sio.seek(0)
    line = Image.open(sio)
    im.paste(line, offset)
    #im.show()
    im.save(imgName)
