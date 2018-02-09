import tensorflow as tf
import numpy as np
import pandas as pd
from tkinter import *
from PIL import ImageTk, Image
import os

imdata = []
spaces = []
image = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG.png"
sentence = Image.open("Sentences/" + image)
width, height = sentence.size
for i in range(0, width+1, 28):
    im = sentence.crop((i, 0, i+28, 28)).rotate(90)
    a = list(map(float, list(im.getdata())))
    if sum(a) == 0:
        spaces.append(i/28)
        continue
    imdata.append(a)

with tf.Session() as session:
    saver = tf.train.import_meta_graph("Saved Models/mymodel.meta")
    saver.restore(session, "Saved Models/mymodel")
    logits = session.run(tf.nn.relu(tf.matmul(imdata, session.run("w1:0")) + session.run("b1:0")))
    logits = session.run(tf.nn.relu(tf.matmul(logits, session.run("w2:0")) + session.run("b2:0")))
    logits = session.run(tf.nn.relu(tf.matmul(logits, session.run("w3:0")) + session.run("b3:0")))
    logits = session.run(tf.nn.softmax(tf.matmul(logits, session.run("w4:0")) + session.run("b4:0")))
    recog = list(map(chr, np.argmax(logits, 1)+65))

[recog.insert(int(x), " ") for x in spaces]
translation = "".join([x for x in recog])

root = Tk()
img = ImageTk.PhotoImage(Image.open("Sentences/" + image))
panel = Label(root, image = img)
text = Text(root)
text.insert(INSERT, translation)
panel.pack(side = "bottom", fill = "both", expand = "yes")
text.pack()
root.mainloop()
