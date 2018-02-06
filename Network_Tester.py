import tensorflow as tf
import numpy as np
import pandas as pd
import random
from PIL import Image
from painter import get_image_src
from scipy import misc

get_image_src()
im = misc.imread("step61.png")
imgray = []
for line in im:
	for pixel in line:
		if np.average(pixel) == 255.0:
			imgray.append(0.0)
		else:
			imgray.append(np.average(pixel))

#inp_img = tf.constant(valid_dataset[0], dtype=tf.float32)
with tf.Session() as session:
	saver = tf.train.import_meta_graph("Saved Models/mymodel.meta")
	saver.restore(session, "Saved Models/mymodel")
	logits = session.run(tf.nn.relu(tf.matmul([imgray], session.run("w1:0")) + session.run("b1:0")))
	logits = session.run(tf.nn.relu(tf.matmul(logits, session.run("w2:0")) + session.run("b2:0")))
	logits = session.run(tf.nn.relu(tf.matmul(logits, session.run("w3:0")) + session.run("b3:0")))
	logits = session.run(tf.nn.softmax(tf.matmul(logits, session.run("w4:0")) + session.run("b4:0")))
	print(chr(np.argmax(logits) + 65)) #, chr(np.argmax(valid_labels[0]) + 65))
