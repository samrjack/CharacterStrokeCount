'''
Downloads a given url, removes all duplicate and non-Chinese characters, and
returns a single string containing everything.
'''
import urllib2
import char_util as cu

'''Journey to the west'''
#DEFAULT_URL = "http://www.gutenberg.org/cache/epub/23962/pg23962.html"

'''Some shorter (but still long) website for testing purposes'''
DEFAULT_URL = "http://www.humancomp.org/unichtm/linjilu8.htm"

'''
get_data
    Loads a given file from a url, removes duplicates, and writes everything
    to a given output file.
'''
def get_data(url=DEFAULT_URL):
    data = urllib2.urlopen(url)
    completeSet = set('')
    for line in data:
        u8Line = unicode(line, "utf8")
        completeSet = completeSet.union(filter(lambda c: cu.get_stroke_count(c) > 0, u8Line))
    data.close()
    return "".join(completeSet)
