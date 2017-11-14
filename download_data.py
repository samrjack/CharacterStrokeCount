import urllib2
import char_util as cu

#DEFAULT_URL = "http://www.gutenberg.org/cache/epub/23962/pg23962.html"
DEFAULT_URL = "http://www.humancomp.org/unichtm/linjilu8.htm"

# map(lambda x: make_char(x, 32), add_T_and_S(get_data()))
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
