'''
Utility functions to get information about a character.
'''
from hanziconv import HanziConv


def trad_and_simp(inputString):
    '''
    Takes in a unicode string containing chinese characters and
    makes sure it contains both traditional and simplified versions
    of every character. If both versions are not present, whatever
    is missing is added. The returned string is in no guarenteed
    order, just guarenteed to have both character sets where possible.

    Parameters
    ----------
    inputString : String
        A string containing traditional and/or simplified
        Chinese characters. These will be expanded so that all simplified
        and traditional characters are present.

    Returns
    -------
    String
        A string is returned that contain traditional and simplified
        versions of every Chinese character found in the input string.
    '''
    totalSet = set(inputString)
    totalSet = totalSet.union(HanziConv.toSimplified(inputString))
    totalSet = totalSet.union(HanziConv.toTraditional(inputString))
    return "".join(totalSet)


def get_stroke_count(char):
    '''
    finds the official stroke count of a given unicode chinese character.

    Parameters
    ----------
    char : Unicode Character
        a single unicode character to be converted to a stroke count.

    Returns
    -------
    int
        Returns the stroke count of the given character. If the character isn't
        recognized, 0 is returned.
    '''
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
