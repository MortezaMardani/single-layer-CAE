# convolutional autoencoder

import numpy as np
import tensorflow as tf
import pdb

# data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# parameters
learning_rate = 0.1
batch_size = 100
display_step = 1
#drop_out = 0.5
num_features = 50

# tf graph input
x = tf.placeholder(tf.float32,[None,784])
#keep_prob = tf.placeholder(tf.float32)

# create model
def conv_net(_x, _weights, _biases):

     #reshape input pictures
     _x = tf.reshape(_x, shape=[-1,28,28,1])
     
     # Encoding
     # Convolution Layer
     conv1 = tf.nn.conv2d(_x, _weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
     conv1 = tf.add(conv1, _biases['bc1'])
     conv1 = tf.nn.relu(conv1)   #[-1,28,28,32]

     #fully connected layer
     dense1 = tf.reshape(conv1, [-1, 28*28*32]) 
     dense1 = tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1'])
     dense1 = tf.nn.relu(dense1)  #[-1,num_features]
     #feature_vec = dense1

     # Decoding
     # fully connected layer
     dense2 = tf.matmul(dense1, _weights['wd2']) #[-1,32*28*28]
     dense2 = tf.reshape(dense2, shape=[-1,28,28,32])
     dense2 = tf.add(dense2, _biases['bd2'])
     dense2 = tf.nn.relu(dense2)
    
     # deconvolution
     recon = tf.nn.conv2d(dense2,_weights['wc2'],strides=[1,1,1,1],padding='SAME')   #shape=[-1,28,28,32]?????
     #include relu
     recon = tf.nn.relu(recon)
     recon_vec = tf.add(tf.reshape(recon, shape=[-1,28*28]), _biases['bc2'])
     return recon_vec

#define variables
weights = {'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])), # 5x5 conv, 1 input, 32 outputs
           'wd1': tf.Variable(tf.random_normal([28*28*32, 50])), # fully connected, 28*28*32 inputs, num_features outputs
           'wd2': tf.Variable(tf.random_normal([50, 28*28*32])), # fully connected, 12*12*32 inputs, num_features outputs
           'wc2': tf.Variable(tf.random_normal([5, 5, 32, 1])), # 5x5 conv, 32 input, 1 outputs
}

biases = {'bc1': tf.Variable(tf.random_normal([32])),
          'bd1': tf.Variable(tf.random_normal([50])),
          'bd2': tf.Variable(tf.random_normal([32])),
          'bc2': tf.Variable(tf.random_normal([1])),
}


#summary ops to collect the data
wc1_hist = tf.histogram_summary("weights",weights['wc1'])
bc1_hist = tf.histogram_summary("biases",biases['bc1'])

#reconstructed image
recon_image_vec = conv_net(x, weights, biases)
recon_image = tf.reshape(recon_image_vec[1,:],shape=[-1,28,28,1])
#x_image = 

#summary of the images
recon_image_summary = tf.image_summary("image_recon",recon_image)
#original_image = tf.image_summary("image_actual",x_image)

#ls cost
cost = tf.reduce_mean(tf.square(recon_image_vec - x))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#summary of cost
cost_summary = tf.scalar_summary("cost",cost)

#merge all summaries
merged = tf.merge_all_summaries()


# Initializing the variables
init = tf.initialize_all_variables()

#launch the graph
sess = tf.Session()
sess.run(init)

writer = tf.train.SummaryWriter("/tmp/mnist_logs", sess.graph.as_graph_def(add_shapes=True))  

for i in range(50):

     #print i
     #print sess.run(cost)

     batch_x = mnist.train.next_batch(batch_size)[0]
     sess.run(optimizer, feed_dict={x:batch_x})
     if i % display_step == 0:

           loss = sess.run(cost, feed_dict={x:batch_x}) 
           summary_str = sess.run(merged, feed_dict={x:batch_x})
           writer.add_summary(summary_str,i)
           print "Iter" + str(i) + ", Cost= " + "{:.6f}".format(loss)


#print sess.run()
print "Finished!"
writer.close()

#tensorboard
#tensorboard --logdir=/tmp/mnist_logs

#remove the folder
#rm -r /tmp/mnist_logs
#local host:6006

#sess.close()
