import tensorflow as tf

print("tensorflow version {}".format(tf.__version__))

from tensorflow.python.keras.models import Model 
from tensorflow.python.keras.layers import Input 
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Activation
from tensorflow.python.keras.layers import MaxPooling2D 




def model_fn(features, labels, mode, params):
	X_input = features["image"]

	# Conv1
	X = Conv2D(
		filters=96, 
		kernel_size=(11, 11), 
		strides=(4, 4), 
		padding="valid", 
		name="Conv_1")(X_input)
	X = Activation("relu", name="Relu_1")(X)

	X = MaxPooling2D(pool_size=3, strides=2, name="MaxPool_1")(X)

	# Conv2
	X = Conv2D(
		filters=256,
		kernel_size=(5, 5), 
		strides=(1, 1), 
		padding="same", 
		name="Conv_2")(X)
	X = Activation("relu", name="Relu_2")(X)

	X = MaxPooling2D(pool_size=3, strides=2, name="MaxPool_2")(X)

	# Conv3 
	X = Conv2D(
		filters=384, 
		kernel_size=(3, 3), 
		strides=(1, 1), 
		padding="same", 
		name="Conv_3")(X)
	X = Activation("relu", name="Relu_3")(X)

	# Conv4
	X = Conv2D(
		filters=384, 
		kernel_size=(3, 3), 
		strides=(1, 1),
		padding="same", 
		name="Conv_4")(X)
	X = Activation("relu", name="Relu_4")(X)

	# Conv5
	X = Conv2D(
		filters=256, 
		kernel_size=(3, 3), 
		strides=(1, 1), 
		padding="same", 
		name="Conv_5")(X)
	X = Activation("relu", name="Relu_5")(X)

	X = MaxPooling2D(pool_size=2, strides=2, name="MaxPool_3")(X)

	# Flatten
	X = Flatten(name="Flatten")(X)

	# Fully connected layer 1
	X = Dense(4096, name="FC_1")(X)

	# Fully connected layer 2 
	X = Dense(4096, name="FC_2")(X)

	# Fully connected layer 3 
	X = Dense(10, name="FC_3")(X)	

	# logits output of the neural network 
	logits = X 

	# Softmax output of the neural network 
	y_pred = tf.nn.softmax(logits=logits)

	# Classification output of the neural network 
	y_pred_cls = tf.argmax(y_pred, axis=1)


	if mode == tf.estimator.ModeKeys.PREDICT:
		# If the estimator is supposed to be in prediction-mode 
		# then use the predicted class-number that is output by 
		# the neural network. Optimization, etc. is not needed. 

		spec = tf.estimator.EstimatorSpec(mode=mode, 
			predictions=y_pred_cls)

	else: 
		# Otherwise the estimator is supposed to be in either training or evaluation-mode. 
		# Note that the loss function is also required in Evaluation mode. 

		# Define the loss-function to be optimized, by first 
		# calculating the cross-entropy between the output of 
		# the neural network and the true labels for the input data. 
		# This gives the cross-entropy for each image in the batch. 

		# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, 
		# 	logits=logits)

		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, 
			logits=logits)

		# Reduce the cross-entropy batch-tensor to a single number 
		# which can be used in optimization of the neural network. 
		loss = tf.reduce_mean(cross_entropy)

		# Define the optimizer for improving the neural network 
		optimizer = getattr(tf.train, params["optimizer"]["optimizer_type"])(
			**params["optimizer"]["optimizer_params"])

		# Get the TensorFlow op for doing a single optimization step 
		train_op = optimizer.minimize(
			loss=loss, global_step=tf.train.get_global_step())

		# Define the evaluation metrics, 
		# in this case the classification accuracy 
		# metrics = {
		# "accuracy": tf.metrics.accuracy(labels, y_pred_cls)
		# }

		metrics = {
		"accuracy": tf.metrics.accuracy(labels, y_pred)
		}

		# Wrap all of this in a n EstimatorSpec
		spec = tf.estimator.EstimatorSpec(
			mode=mode, 
			loss=loss, 
			train_op=train_op, 
			eval_metric_ops=metrics)

	return spec



class AlexNet():
	""" Create AlexNet class """ 
	# def __init__(self, config):
	# 	self.config = config 
	# 	self.model = None 	

	def __init__(self):
		# self.config = config 
		self.model = None 


	# def construct_model(self, input_shape=(224, 224, 3)):
	def construct_model(self, features):
		""" Create model architecture """
			
		# Define the input
		# X_input = Input(shape=input_shape, name="Input")

		X = features['image']

		# Conv1
		X = Conv2D(
			filters=96, 
			kernel_size=(11, 11), 
			strides=(4, 4), 
			padding="valid", 
			name="Conv_1")(X_input)
		X = Activation("relu", name="Relu_1")(X)

		X = MaxPooling2D(pool_size=3, strides=2, name="MaxPool_1")(X)

		# Conv2
		X = Conv2D(
			filters=256,
			kernel_size=(5, 5), 
			strides=(1, 1), 
			padding="same", 
			name="Conv_2")(X)
		X = Activation("relu", name="Relu_2")(X)

		X = MaxPooling2D(pool_size=3, strides=2, name="MaxPool_2")(X)

		# Conv3 
		X = Conv2D(
			filters=384, 
			kernel_size=(3, 3), 
			strides=(1, 1), 
			padding="same", 
			name="Conv_3")(X)
		X = Activation("relu", name="Relu_3")(X)

		# Conv4
		X = Conv2D(
			filters=384, 
			kernel_size=(3, 3), 
			strides=(1, 1),
			padding="same", 
			name="Conv_4")(X)
		X = Activation("relu", name="Relu_4")(X)

		# Conv5
		X = Conv2D(
			filters=256, 
			kernel_size=(3, 3), 
			strides=(1, 1), 
			padding="same", 
			name="Conv_5")(X)
		X = Activation("relu", name="Relu_5")(X)

		X = MaxPooling2D(pool_size=2, strides=2, name="MaxPool_3")(X)

		# Flatten
		X = Flatten(name="Flatten")(X)

		# Fully connected layer 1
		X = Dense(4096, name="FC_1")(X)

		# Fully connected layer 2 
		X = Dense(4096, name="FC_2")(X)

		# Fully connected layer 3 
		X = Dense(10, name="FC_3")(X)

		# Create model 
		model = Model(inputs=X_input, outputs=X, name="AlexNet")

		return model


if __name__ =='__main__':
	net = AlexNet()
	model = net.construct_model()

	print("model summary : ", model.summary())


