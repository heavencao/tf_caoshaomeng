from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import input_datas
# Import MNIST data
trainx,trainy=input_datas.data_sets()

logs_path = '/tmp/tensorflow_logs/'

testx=trainx[2000:2050]
testy=trainy[2000:2050]

learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 19 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None,440,440,1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 440, 440, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([110*110*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
with tf.name_scope('Model'):	
    pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
with tf.name_scope('Loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
with tf.name_scope('SGD'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
with tf.name_scope('Accuracy'):
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()
saver=tf.train.Saver()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    #saver.restore(sess,"./models/tensorflow_rnn_mode.ckpt")
    j=0
    for i in range(100000):
     
    	batchx =trainx[j*100:int(j+1)*100]
    	batchy =trainy[j*100:int(j+1)*100]

	if True:
	    j += 1

    	    _, loss, acc, summary= sess.run([optimizer,cost, accuracy, merged_summary_op], feed_dict={x: batchx, y: batchy, keep_prob: 1.0})

            print(" Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
    	    
		  
            summary_writer.add_summary(summary,i)
  	if j==22 :
	    print('___________________________________')
	    
    	    batchx =trainx[j*100:]
    	    batchy =trainy[j*100:]
	    
    	    loss, acc = sess.run([cost, accuracy], feed_dict={x: batchx,
                                                              y: batchy,
                                                              keep_prob: 1.0})
           
            print(" Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

	    print("----------------------------------")
	    j=0
    saver.save(sess,"./models/tensorflow_rnn_mode.ckpt")
'''
 
# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 19 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, 440,440,1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
   
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 440, 440, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    fc1 = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    out=tf.nn.softmax(fc1)
    return out,fc1

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([110*110*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, 19]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([19]))
}

# Construct model

pred,fc = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = -tf.reduce_mean(y*tf.log(pred))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


t_y=tf.argmax(pred, 1)
p_y=tf.argmax(y,1)
c_p=tf.equal(t_y,p_y)







# Initializing the variables
init = tf.global_variables_initializer()

j=0

tf.summary.scalar('loss',cost)
tf.summary.scalar('acc',accuracy)
merged_summary_op=tf.summary.merge_all()

sess=tf.Session()
summary_writer=tf.summary.FileWriter('/tmp/logdir',sess.graph)
# Launch the graph
saver=tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    
    
    
    #saver.restore(sess,"./models/rnnmode.ckpt")

    for i in range(100000):
     
    	batchx =trainx[j*100:int(j+1)*100]
    	batchy =trainy[j*100:int(j+1)*100]

  	if j<=20:
    	    j += 1
	    sess.run(optimizer,feed_dict={x:batchx,y:batchy,keep_prob:0.75})
  	else :
	    j=0
    	    batchx =trainx[j*100:]
    	    batchy =trainy[j*100:]
    	    loss, acc,pred_y,a,b,c = sess.run([cost, accuracy,pred,t_y,p_y,fc], feed_dict={x: batchx,
                                                              y: batchy,
                                                              keep_prob: 1.0})

            print(" Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

    	    print (a)
    	    print ('---------')
    	    print (b)
    	    print (c)


	    
    #saver.save(sess,"./models/rnnmode.ckpt")
    # Calculate accuracy for 256 mnist test images
   


    
'''


   



    
