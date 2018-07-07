import tensorflow as tf


# def compute_metrics(logits, targets):
# 	""" Regroup all computed metrics """
# 	with tf.name_scope("metric"):
# 		cross_entropy_mean = compute_loss_xent(logits, targets)
# 		accuracy = compute_accuracy(logits, targets)

# 		return cross_entropy_mean, accuracy 

def compute_loss_xent(logits, targets):
	""" Compute cross entropy as our loss function """
	with tf.name_scope("cross_entropy"):
		# Get rid of extra dimensions and cast targets into integers
		targets = tf.squeeze(tf.cast(targets, tf.int32))
		# Calculate cross entropy from logits and targets
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2( \
			logits=logits, labels=targets)
		# Take the average loss across batch size
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")

		return cross_entropy_mean

def compute_accuracy(logits, targets):
	""" Compute the accuracy """
	with tf.name_scope("accuracy"):
		correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(targets, 1)) # or tf.nn.in_top_k(logits, targets, 1)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		return accuracy