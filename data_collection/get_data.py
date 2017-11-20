'''
Main file for producing data
'''
import image_maker as im
import char_util as cu
import download_data as d
import time

ac = cu.trad_and_simp(d.get_data())

def main():
    # Only by adding count would the program properly work
    # Don't know why, but will leave it as is for now.
    count = 0
    for c in ac[count:]:
        print c + " " + str(count)
        count += 1
        im.make_char(c,32)
        time.sleep(.01)

    # Using map would crash the program for some reason
    # map(lambda x: im.make_char(x, 32), cu.trad_and_simp(d.get_data()))

if __name__ == "__main__":
    main()

