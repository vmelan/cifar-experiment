import tensorflow as tf 


def conv2d(x, W, b, strides=1, name="conv"):
	""" Convolution layer """
	with tf.name_scope(name):
		x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
		x = tf.nn.bias_add(x, b)

		return tf.tanh(x)

def maxpool2d(x, strides=2, name="maxpool"):
	""" Pooling layer """
	with tf.name_scope(name):
		return tf.nn.max_pool(x, ksize=[1, strides, strides, 1], \
			strides=[1, strides, strides, 1], padding='SAME')

def fully_connected(x, W, b, act=True, name="fc"):
	""" Fully connected layer """
	with tf.name_scope(name):
		x = tf.add(tf.matmul(x, W), b)

		if act:
			return tf.tanh(x)
		else:
			return x 

