'''
adapted from
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import numpy as np
import sys
import tensorflow as tf
from ROOT import TCanvas, TH1F, TH2F

# this function handles batching for the dataset, sim. to mnist's in-built
def next_batch(num, data, labels):
    '''
    Return a total of num random samples and labels.
    '''
    nev, nfe = data.shape
    idx = np.arange(0 , nev)
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def read_in_dataset(fname,n_class):
    # read in dataset
    data_file = open(fname, 'r')
    features_arr, labels_arr = [], []
    header_arr = data_file.readline().split(',')
    print("reading in dataset from " + fname)
    n_feats = len(header_arr) - 1
    print("found " + str(n_feats) + " features")
    lin = data_file.readline()
    n_evts = 0
    while len(lin) > 2: # and n_evts < 10000:
        n_evts += 1
        arr = lin.split(',')
        labels_arr.append([int(arr[0]), 1-int(arr[0])])
        for i in range(len(arr) - 1):
            arr[i+1] = float(arr[i+1])
        arr.pop(0)
        features_arr.append(arr)
        lin = data_file.readline()

    features_arr = np.array(features_arr)
    labels_arr = np.array(labels_arr)
    data_file.close()
    return n_feats, header_arr, features_arr, labels_arr

n_classes = 2 # bkgd = 0, sig = 1
n_feats, Harr, Xarr, Yarr = read_in_dataset(sys.argv[1], n_classes)

# Parameters
learning_rate = 0.1
n_steps = 100
batch_size = 1000
display_step = 100

# Network Parameters
n_hidden_1 = 20 # 1st layer number of neurons
n_hidden_2 = 20 # 2nd layer number of neurons
n_input = n_feats

print("features array has dimension " + str(Xarr.shape))
print("taraget array has dimension " + str(Yarr.shape))

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1' : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with n_hidden_1 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with n_hidden_2 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for step in range(1, n_steps+1):
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x, batch_y = next_batch(batch_size, Xarr, Yarr)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.4f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy
    n_feats, Harr, Xarr, Yarr = read_in_dataset(sys.argv[2], n_classes)
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: Xarr,
                                      Y: Yarr}))


    prediction2 = tf.nn.softmax(logits)
    test_pred = sess.run([prediction2], feed_dict={X: Xarr,
                                                  Y: Yarr})
    print(test_pred)

    # h = TH1F('h','h',100,0,1000)
    # for i in range(80000):
    #     print(test_pred[i][0])
    #     #h.Fill(test_pred[i][0])
    # can = TCanvas('can','can',20,20,800,700)
    # can.cd(1)

# if __name__ == '__main__':
#   rep = ''
#   while not rep in [ 'q', 'Q' ]:
#     rep = input( 'enter "q" to quit: ' )
#     if 1 < len(rep):
#       rep = rep[0]
