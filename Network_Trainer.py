import tensorflow as tf
import pandas as pd
import numpy as np
import random

num_labels = 26
train_size = 88000
valid_size = 10000
num_fields = 784
batch_size = 100
beta = 0.02
num_nodes_1 = 405
num_nodes_2 = int(num_nodes_1 * 0.5)
num_nodes_3 = int(num_nodes_1 * np.power(0.5, 2))
learn_rate = 0.00001

def abcd():
    data = pd.read_csv("emnist-letters-train.csv")
    data = data.as_matrix()
    return data
train_data = abcd()
train_dataset = train_data[:, 1:]
train_labels = train_data[:, :1]
a = []
for i in train_labels:
    b = [0.0] * num_labels
    b[int(i[0]) - 1] = 1.0
    a.append(b)
train_labels = np.array(a)

def bcdf():
    data1 = pd.read_csv("emnist-letters-test.csv")
    data1 = data1.as_matrix()
    return data1
valid_data = bcdf()
valid_dataset = valid_data[:, 1:]
valid_labels = valid_data[:, :1]
a = []
for i in valid_labels:
    b = [0.0] * num_labels
    b[int(i[0]) - 1] = 1.0
    a.append(b)
valid_labels = np.array(a)

del valid_data
del train_data

tf_train_data = tf.placeholder(tf.float32, shape=[batch_size, num_fields])
tf_train_labels = tf.placeholder(tf.float32, shape=[batch_size, num_labels])
tf_valid_data = tf.constant(valid_dataset, dtype=tf.float32)

weights_1 = tf.Variable(tf.truncated_normal(shape=[num_fields, num_nodes_1], stddev=np.sqrt(2 / num_fields)), name="w1")
biases_1 = tf.Variable(tf.zeros([num_nodes_1]), name="b1")
weights_2 = tf.Variable(tf.truncated_normal(shape=[num_nodes_1, num_nodes_2], stddev=np.sqrt(2 / num_nodes_1)), name="w2")
biases_2 = tf.Variable(tf.zeros([num_nodes_2]), name="b2")
weights_3 = tf.Variable(tf.truncated_normal(shape=[num_nodes_2, num_nodes_3], stddev=np.sqrt(2 / num_nodes_2)), name="w3")
biases_3 = tf.Variable(tf.zeros([num_nodes_3]), name="b3")
weights_4 = tf.Variable(tf.truncated_normal(shape=[num_nodes_3, num_labels], stddev=np.sqrt(2 / num_nodes_3)), name="w4")
biases_4 = tf.Variable(tf.zeros([num_labels]), name="b4")

logits = tf.nn.relu(tf.matmul(tf_train_data, weights_1) + biases_1)
logits = tf.nn.relu(tf.matmul(logits, weights_2) + biases_2)
logits = tf.nn.relu(tf.matmul(logits, weights_3) + biases_3)
logits = tf.matmul(logits, weights_4) + biases_4
train_predictions = tf.nn.softmax(logits, name="train_predictions")

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
reg = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(weights_3) + tf.nn.l2_loss(weights_4)
loss = tf.reduce_mean(loss + reg * beta)
optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss)

logits = tf.nn.relu(tf.matmul(tf_valid_data, weights_1) + biases_1)
logits = tf.nn.relu(tf.matmul(logits, weights_2) + biases_2)
logits = tf.nn.relu(tf.matmul(logits, weights_3) + biases_3)
logits = tf.matmul(logits, weights_4) + biases_4
valid_predictions = tf.nn.softmax(logits, name="valid_predictions")

saver = tf.train.Saver()

num_steps = 8000
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

with tf.Session() as session:
    try:
        saver.restore(session, "C:/Users/Vedang Naik/Desktop/Programming/Handwriting Recognition/Saved Models/mymodel")
        print("Found file.")
    except:
        tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        i = random.randint(0, train_size-batch_size)
        batch_data = train_dataset[i:i+batch_size]
        batch_labels = train_labels[i:i+batch_size]

        feed_dict = {tf_train_data: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_predictions], feed_dict=feed_dict)

        if step % 1000 == 0:
            print("Minibatch loss at step {}: {}".format(step, l))
            print("Minibatch accuracy: {:.1f}".format(accuracy(predictions, batch_labels)))
            print("Validation accuracy: {:.1f}".format(accuracy(valid_predictions.eval(), valid_labels)))
    saver.save(session, "C:/Users/Vedang Naik/Desktop/Programming/Handwriting Recognition/emnist/mymodel")
