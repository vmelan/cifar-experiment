import tensorflow as tf 

# class LeNet(object):
# 	"""
# 	LeNet model
# 	"""

# 	def __init__(self):
# 		""" Initialize parameters of the class """
	
# 		self.input_height, self.input_width = 32, 32, 3
# 		self.filter_height, self.filter_width = 3, 3
# 		self.num_classes = 10
# 		self.depth_in = 3
# 		self.depth_out_1 = 6
# 		self.depth_out_2 = 16
# 		self.depth_out_3 = 120

# 		self.define_params()

# 	def define_params(self):
# 		""" Define the weights and the bias """

# 		self.weights = { 
# 		'wc1': tf.Variable(tf.truncated_normal([filter_height, filter_width, depth_in, depth_out_1])), \
# 		'wc2': tf.Variable(tf.truncated_normal([filter_height, filter_width, depth_out_1, depth_out_2])), \
# 		}

# 		self.bias = {
# 		'bc1': tf.Variable(tf.constant(0, shape=[depth_out_1])), \
# 		'bc2': tf.Variable(tf.constant(0, shape=[depth_out_2])), \
# 		'bc3': tf.Variable(tf.constant(0, shape=[depth_out_3])), \

# 		}

def forward(self, x):
	""" Forward pass of LeNet given an input image x """

	# Convolutional layer 1
	self.conv_1 = self.conv2d(x, self.weights["wc1"], self.bias["bc1"], name="conv_1")
	self.maxpool_1 = self.maxpool2d(self.conv_1, strides=2, name="maxpool_1")
	# Convolutional layer 2
	self.conv_2 = self.conv2d(self.maxpool_1, self.weights["wc2"], self.bias["bc2"], name="conv_2")
	self.maxpool_2 


# def conv2d(x, W, b, strides=1, name="conv"):
# 	""" Convolution layer """
# 	with tf.name_scope(name):
# 		x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
# 		x = tf.nn.bias_add(x, b)

# 		return tf.tanh(x)

def conv2d(input, num_input_channels, num_filters, filter_shape, strides=[1, 1], name="conv"):
	""" Convolution layer """
	with tf.name_scope(name):
		w = tf.Variable(tf.truncated_normal([filter_shape[0], filter_shape[1], num_input_channels, num_filters], \
			stddev=0.02), name=name+"_W")
		b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name=name+"_b")

		# conv, bias and then activation
		output = tf.nn.conv2d(input, w, strides=[1, strides[0], strides[1], 1], padding='VALID')
		output = tf.nn.bias_add(conv, b)
		output = tf.tanh(output)

		return output

def avgpool2d(x, strides=[2, 2], name="avgpool"):
	""" Average Pooling layer """
	with tf.name_scope(name):
		return tf.nn.pool(x, pooling_type='AVG', \
			window_shape=[1, strides[0], strides[1], 1], \
			strides=[1, strides[0], strides[1], 1], padding='VALID', \
			name=name)

# def fully_connected(x, W, b, act=True, name="fc"):
# 	""" Fully connected layer """
# 	with tf.name_scope(name):
# 		x = tf.add(tf.matmul(x, W), b)

# 		if act: # if True, use tanh activation function, otherwise do not apply any activation
# 			return tf.tanh(x)
# 		else:
# 			return x 

def fully_connected(input, neurons_in, neurons_out, act=True, name="fc"):
	""" Fully connected layer """
	with tf.name_scope(name):
		w = tf.Variable(tf.truncated_normal([neurons_in, neurons_out], stddev=0.02), name=name+"_W")
		b = tf.Variable(tf.constant(0.1, shape=[neurons_out]), name=name+"_b")

		output = tf.add(tf.matmul(input, W), b)

		if act: # if True, use tanh activation function, otherwise do not apply any activation
			return tf.tanh(output)
		else:
			return output

# def compute_loss(logits, targets):
# 	""" Compute cross entropy as our loss function """
# 	with tf.name_scope("cross_entropy"):
# 		# Get rid of extra dimensions and cast targets into integers
# 		targets = tf.squeeze(tf.cast(targets, tf.int32))
# 		# Calculate cross entropy from logits and targets
# 		cross_entropy = tf.nn.softmax_cross_entropy_with_logits( \
# 			logits=logits, labels=targets)
# 		# Take the average loss across batch size
# 		cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")

# 		return(cross_entropy_mean)

# def train_step(loss_value, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08):
# 	""" Use an AdamOptimizer to train the network """
# 	with tf.name_scope("train"):
# 		# Create optimizer
# 		my_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, \
# 			beta2=beta2, epsilon=epsilon)
# 		# Initialize train step
# 		train_step = my_optimizer.minimize(loss_value)

# 		return train_step

# def compute_accuracy(logits, targets):
# 	""" Compute the accuracy """
# 	with tf.name_scope("accuracy"):
# 		correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(targets, 1))
# 		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 		return accuracy

