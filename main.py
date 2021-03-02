from utils import *

## crop images to remove whitespace surrounding the pciture.
for im in glob.glob("/Users/huynslol/Documents/stamtest/*.NEF"):
    coinCrop(im, r"/Users/huynslol/Documents/test")
    # coinReader(im)


