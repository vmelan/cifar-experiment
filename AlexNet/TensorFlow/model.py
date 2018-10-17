import tensorflow as tf

print("tensorflow version {}".format(tf.__version__))

from tensorflow.python.keras.models import Model 
from tensorflow.python.keras.layers import Input 
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Activation
from tensorflow.python.keras.layers import MaxPooling2D 


class AlexNet():
	""" Create AlexNet class """ 
	# def __init__(self, config):
	# 	self.config = config 
	# 	self.model = None 	

	def __init__(self):
		# self.config = config 
		self.model = None 


	def construct_model(self, input_shape=(224, 224, 3)):
		""" Create model architecture """
			
		# Define the input
		X_input = Input(shape=input_shape, name="Input")

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