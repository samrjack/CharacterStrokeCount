'''
Useful utility functions to get information about a character.
'''
from hanziconv import HanziConv


'''
add_T_and_S
    Takes in a unicode string containing chinese characters and
    makes sure it contains both traditional and simplified versions
    of every character. If both versions are not present, whatever
    is missing is added. The returned string is in no guarenteed
    order, just guarenteed to have both character sets where possible.

    inputString - A string containing traditional and/or simplified
        Chinese characters. These will be expanded so that all simplified
        and traditional characters are present.

    returns - A string is returned that contain traditional and simplified
        versions of every Chinese character found in the input string.
'''
def trad_and_simp(inputString):
    totalSet = set(inputString)
    totalSet = totalSet.union(HanziConv.toSimplified(inputString))
    totalSet = totalSet.union(HanziConv.toTraditional(inputString))
    return "".join(totalSet)


'''
get_stroke_count
    finds the official stroke count of a given unicode chinese character.

    char - a single character to be converted to a stroke count.

    returns - The stroke count if one is found. 0 if no the character isn't
        recognized.
'''
def get_stroke_count(char):
    # Gets the hex number withou 0x at beginning as is found in the
    # source file.
    hexNum = hex(ord(char))[2:].upper()
    return int(_strokeCount.get(hexNum, '0'))


'''
Read in the stroke counts of all chinese unicode characters from the
totalstrokes file whenever this file is loaded. Values can be accessed
using the get_stroke_count method.
'''
_strokeCount = {}
with open("count/totalstrokes.txt") as f:
    for line in f:
        (key, val) = line.split()
        _strokeCount[key] = val
