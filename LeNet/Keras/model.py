from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, Activation, AveragePooling2D
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.models import model_from_json
import keras.backend as K
import keras.optimizers as optim

# def gaussian(x):
# 	return K.exp(-K.pow(x, 2))

class LeNet(object):
	""" Create LeNet class """
	def __init__(self, config):
		self.config = config
		self.model = None

		# Get optimizer
		self.optimizer = getattr(optim, config["optimizer"]["optimizer_type"])
		self.optimizer = self.optimizer(**config["optimizer"]["optimizer_params"])

		self.build_model()

	def build_model(self, input_shape=(32, 32, 1)):
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
		X = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), \
			kernel_initializer=glorot_uniform(seed=0), \
			padding="valid", \
			name="Conv_2")(X)		
		X = Activation("tanh", name="Tanh_2")(X)
		X = AveragePooling2D(pool_size=(2, 2), name="AvgPool_2")(X)
		# 3rd conv layer : CONV + TANH
		X = Conv2D(filters=120, kernel_size=(5, 5), strides=(1, 1), \
			kernel_initializer=glorot_uniform(seed=0), \
			padding="valid", \
			name="Conv_3")(X)
		X = Activation("tanh", name="Tanh_3")(X)		
		# Flatten
		X = Flatten(name="Flatten")(X)
		# Fully connected layer 1 : DENSE + TANH
		X = Dense(84, name="FC_1")(X)
		X = Activation("tanh", name="Tanh_4")(X)

		# Output with Gaussian connections
		X = Dense(10, name="FC_2")(X)

		X = Activation("softmax", name="softmax")(X)

		# X = Activation(gaussian, name="rbf")(X)

		# Create model
		self.model = Model(inputs=X_input, outputs=X, name="LeNet")

		# Compile model
		self.compile()

	def compile(self):
		""" 
		Compile model using Adam optimizer
		"""
		self.model.compile(optimizer=self.optimizer, 
			loss='categorical_crossentropy', 
			metrics=['accuracy'])		
