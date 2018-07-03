import tensorflow as tf 

def conv2d(input, num_input_channels, num_filters, filter_shape=[5, 5], strides=[1, 1], name="conv"):
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

def avgpool2d(input, filter_shape=[2, 2], strides=[2, 2], name="avgpool"):
	""" Average Pooling layer """
	with tf.name_scope(name):
		return tf.nn.pool(input, pooling_type='AVG', \
			window_shape=[1, filter_shape[0], filter_shape[1], 1], \
			strides=[1, strides[0], strides[1], 1], padding='VALID', \
			name=name)


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

def LeNet(input_image):

	# 1st conv layer : CONV + TANH + AVERAGE POOL 
	conv_1 = conv2d(input_image, num_input_channels=3, num_filters=6, \
		filter_shape=[5, 5], strides=[1, 1], name="conv1")
	avgpool_1 = avgpool2d(conv_1, filter_shape=[2, 2], \
		strides=[2, 2], name="avgpool1")

	# 2nd conv layer : CONV + TANH + AVERAGE POOL
	conv_2 = conv2d(avgpool_1, num_input_channels=6, num_filters=16, \
		filter_shape=[5, 5], strides=[1, 1], name="conv_2")
	avgpool_2 = avgpool2d(conv_2, filter_shape=[2, 2], \
		strides=[2, 2], name="avgpool2")

	# 3rd conv layer : CONV + TANH
	conv_3 = conv2d(avgpool_2, num_input_channels=16, num_filters=120, \
		filter_shape=[5, 5], strides=[1, 1], name="conv3")

	# Flatten 
	flattened = tf.reshape(conv_3, [-1, 1 * 1 * 120])

	# Fully connected layer 1 : DENSE + TANH
	fc_1 = fully_connected(flattened, 120, 84, act=True, name="fc1")

	# Output, Fully connected layer 2 : DENSE + RBF
	fc_2 = fully_connected(fc_1, 84, 10, act=False, name="fc2")
	output = tf.exp(-tf.pow(fc_2, 2)) 

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

