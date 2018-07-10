import warnings
warnings.filterwarnings("ignore")


from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, Activation, AveragePooling2D
from keras.models import Model
from keras.initializers import glorot_uniform
import keras.backend as K

class LeNet(object):
	""" Create LeNet class """
	def __init__(self, config):
		self.config = config
		self.model = None

	def build_model(self, input_shape=(32, 32, 3)):
		""" Create model architecture """

		# Define the input as a tensor with shape input_shape 
		X_input = Input(shape=input_shape, name="Input")

		# 1st conv layer : CONV + TANH + AVERAGE POOL
		X = Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), \
			kernel_initializer=glorot_uniform(seed=0), \
			padding="valid", \
			name="Conv_1")(X_input)
		X = Activation("tanh", name="Tanh_1")(X)
		X = AveragePooling2D(pool_size=(2, 2), name="AvgPool_1")(X)
		# 2nd conv layer : CONV + TANH + AVERAGE POOL
		X = Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), \
			kernel_initializer=glorot_uniform(seed=0), \
			padding="valid", \
			name="Conv_2")(X_input)		
		X = Activation("tanh", name="Tanh_2")(X)
		X = AveragePooling2D(pool_size=(2, 2), name="AvgPool_2")(X)
		# 3rd conv layer : CONV + TANH
		X = Conv2D(filters=120, kernel_size=(5, 5), strides=(1, 1), \
			kernel_initializer=glorot_uniform(seed=0), \
			padding="valid", \
			name="Conv_3")(X_input)
		X = Activation("tanh", name="Tanh_3")(X)		
		# Flatten
		X = Flatten(name="Flatten")(X)
		# Fully connected layer 1 : DENSE + TANH
		X = Dense(84, name="FC_1")(X)
		X = Activation("tanh", name="Tanh_4")(X)

		# Output with Gaussian connections
		X = Dense(10, name="FC_2")(X)

		def gaussian(x):
			return K.exp(-K.pow(x, 2))

		X = Activation(gaussian, name="rbf")(X)

		# Create model
		model = Model(inputs=X_input, outputs=X, name="LeNet")

		return model 



# class LeNet:
# 	@staticmethod
# 	def build(width, height, depth, classes):
# 		# initialize the model
# 		model = Sequential()
# 		inputShape = (height, width, depth)

# 		# if we are using "channels first", update the input shape
# 		if K.image_data_format() == "channels_first":
# 			inputShape = (depth, height, width)

# 		# first set of CONV => RELU => POOL layers
# 		model.add(Conv2D(20, (5, 5), padding="same",
# 			input_shape=inputShape))
# 		model.add(Activation("relu"))
# 		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 		# second set of CONV => RELU => POOL layers
# 		model.add(Conv2D(50, (5, 5), padding="same"))
# 		model.add(Activation("relu"))
# 		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 		# first (and only) set of FC => RELU layers
# 		model.add(Flatten())
# 		model.add(Dense(500))
# 		model.add(Activation("relu"))

# 		# softmax classifier
# 		model.add(Dense(classes))
# 		model.add(Activation("softmax"))

# 		# return the constructed network architecture
# 		return model