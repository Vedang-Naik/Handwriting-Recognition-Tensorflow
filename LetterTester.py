import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os

imdata = []

letter = 'E'
directory = "C:/Users/Vedang Naik/Desktop/Programming/Handwriting Recognition/Test Images"
for filename in os.listdir(directory):
    if filename[0] == letter:
        im = Image.open(os.path.join(directory, filename))
        im = im.rotate(90)
        a = list(map(float, list(im.getdata())))
        imdata.append(a)
    else:
        continue

with tf.Session() as session:
    saver = tf.train.import_meta_graph("Saved Models/mymodel.meta")
    saver.restore(session, "Saved Models/mymodel")
    logits = session.run(tf.nn.relu(tf.matmul(imdata, session.run("w1:0")) + session.run("b1:0")))
    logits = session.run(tf.nn.relu(tf.matmul(logits, session.run("w2:0")) + session.run("b2:0")))
    logits = session.run(tf.nn.relu(tf.matmul(logits, session.run("w3:0")) + session.run("b3:0")))
    logits = session.run(tf.nn.softmax(tf.matmul(logits, session.run("w4:0")) + session.run("b4:0")))
    print(letter + ": ", end="")
    print(list(map(chr, np.argmax(logits, 1)+65)).index(letter))
