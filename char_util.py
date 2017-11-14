
from hanziconv import HanziConv


'''
add_T_and_S
    Takes in a unicode string containing chinese characters and adds
    both simplified and traditional characters (without repeat).

inputString - St
'''
def trad_and_simp(inputString):
    totalSet = set(inputString)
    totalSet = totalSet.union(HanziConv.toSimplified(inputString))
    totalSet = totalSet.union(HanziConv.toTraditional(inputString))
    return "".join(totalSet)

'''
Read in the stroke counts of all chinese unicode characters.
'''
strokeCount = {}
with open("count/totalstrokes.txt") as f:
    for line in f:
        (key, val) = line.split()
        strokeCount[key] = val

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
