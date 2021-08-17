from utils import *

## crop images to remove whitespace surrounding the pciture.
for im in glob.glob(r"C:\Users\gelderch\Desktop\sdfdf\*.JPG"):
    coinCrop(im, r"C:\Users\gelderch\Desktop\sdfdf\crop")
    # coinReader(im)


