import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image

imdict = {
    'A': "A0.png",
    'B': "B800.png",
    'C': "C1684.png",
    'D': "D2463.png",
    'E': "E3199.png",
    'F': "F4004.png",
    'G': "G4824.png",
    'H': "H5599.png",
    'I': "I6399.png",
    'J': "J7201.png",
    'K': "K8674.png",
    'L': "L8804.png",
    'M': "M9599.png",
    'N': "N10399.png",
    'O': "O11200.png",
    'P': "P12093.png",
    'Q': "Q12800.png",
    'R': "R13677.png",
    'S': "S14668.png",
    'T': "T65000.png",
    'U': "U65022.png",
    'V': "V65016.png",
    'W': "W65040.png",
    'X': "X65003.png",
    'Y': "Y65064.png",
    'Z': "Z65287.png",
    "space": "space.png"
}

def sentencetoPicture(sentence, imdict):
    result = Image.new("L", (28*len(sentence), 28))
    for i in range(len(sentence)):
        if sentence[i] == " ":
            im = Image.open("Test Images/" + imdict["space"])
        else:
            im = Image.open("Test Images/" + imdict[sentence[i]])
            result.paste(im=im, box=(28*i, 0))
    result.save("Sentences/Question2.png")

sentencetoPicture("CAN MACHINES THINK", imdict)
