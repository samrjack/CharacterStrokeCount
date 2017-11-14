import image_maker as im
import char_util as cu
import download_data as d

map(lambda x: im.make_char(x, 32), cu.trad_and_simp(d.get_data()))
