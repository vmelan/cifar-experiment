import tensorflow as tf 


class LeNet():
	def __init__(self, config):
		# Load config file
		self.config = config 
		# Create saver
		# self.init_saver()
		# Retrieve placeholders
		# self.X, self.y = self.create_placeholder()

	def forward(self, input_image, name="LeNet"):

		with tf.name_scope(name):
			# 1st conv layer : CONV + TANH + AVERAGE POOL 
			conv_1 = self.conv2d(input_image, num_input_channels=1, num_filters=6, \
				filter_shape=[5, 5], strides=[1, 1], name="conv1")
			avgpool_1 = self.avgpool2d(conv_1, filter_shape=[2, 2], \
				strides=[2, 2], name="avgpool1")

			# 2nd conv layer : CONV + TANH + AVERAGE POOL
			conv_2 = self.conv2d(avgpool_1, num_input_channels=6, num_filters=16, \
				filter_shape=[5, 5], strides=[1, 1], name="conv_2")
			avgpool_2 = self.avgpool2d(conv_2, filter_shape=[2, 2], \
				strides=[2, 2], name="avgpool2")

			# 3rd conv layer : CONV + TANH
			conv_3 = self.conv2d(avgpool_2, num_input_channels=16, num_filters=120, \
				filter_shape=[5, 5], strides=[1, 1], name="conv3")

			# Flatten 
			flattened = tf.reshape(conv_3, [-1, 1 * 1 * 120])

			# Fully connected layer 1 : DENSE + TANH
			fc_1 = self.fully_connected(flattened, 120, 84, act=True, name="fc1")

			# Output, Fully connected layer 2 : DENSE + RBF
			fc_2 = self.fully_connected(fc_1, 84, 10, act=False, name="fc2")

			output = fc_2 
			# output = rbf(fc_2, name="rbf") 

			return output 

	def train_optimizer(self, loss_value, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08):
		""" Use an AdamOptimizer to train the network """
		with tf.name_scope("optimizer"):
			# Create optimizer
			my_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, \
				beta2=beta2, epsilon=epsilon)
			# Initialize train step
			train_step = my_optimizer.minimize(loss_value)

			return train_step


	# def create_saver(self):
	# 	""" Tensorflow saver that will be used to save and load the model"""
	# 	self.saver = tf.train.Saver(max_to_keep=self.config["max_to_keep"])
	# 	return self.saver

	# def create_placeholder(self):
	# 	""" Create the placeholders """ 
	# 	X = tf.placeholder(tf.float32, [None, 32, 32, 1])
	# 	y = tf.placeholder(tf.float32, [None, 10])

	# 	return X, y


	def conv2d(self, input, num_input_channels, num_filters, filter_shape=[5, 5], strides=[1, 1], name="conv"):
		""" Convolution layer """
		with tf.name_scope(name):
			w = tf.Variable(tf.truncated_normal([filter_shape[0], filter_shape[1], num_input_channels, num_filters], \
				stddev=0.02), name=name+"_W")
			b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name=name+"_b")

			# conv, bias and then activation
			output = tf.nn.conv2d(input, w, strides=[1, strides[0], strides[1], 1], padding='VALID')
			output = tf.nn.bias_add(output, b)
			output = tf.tanh(output)

			# Histogram summaries for Tensorboard
			tf.summary.histogram("weights", w)
			tf.summary.histogram("biases", b)
			tf.summary.histogram("activations", output)

			return output


	def avgpool2d(self, input, filter_shape=[2, 2], strides=[2, 2], name="avgpool"):
		""" Average Pooling layer """
		with tf.name_scope(name):
			return tf.nn.pool(input, pooling_type='AVG', \
				window_shape=[filter_shape[0], filter_shape[1]], \
				strides=[strides[0], strides[1]], padding='VALID', \
				name=name)


	def fully_connected(self, input, neurons_in, neurons_out, act=True, name="fc"):
		""" Fully connected layer """
		with tf.name_scope(name):
			W = tf.Variable(tf.truncated_normal([neurons_in, neurons_out], stddev=0.02), name=name+"_W")
			b = tf.Variable(tf.constant(0.1, shape=[neurons_out]), name=name+"_b")

			# Histogram summaries for Tensorboard
			tf.summary.histogram("weights", W)
			tf.summary.histogram("biases", b)

			output = tf.add(tf.matmul(input, W), b)

			if act: # if True, use tanh activation function, otherwise do not apply any activation
				return tf.tanh(output)
			else:
				return output


