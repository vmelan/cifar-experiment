import json
from data_loader import DataLoader 
import model
from metrics import compute_loss_xent, compute_accuracy
import tensorflow as tf

def main():
	# Reset graph
	tf.reset_default_graph()


	with open("config.json", "r") as f:
		config = json.load(f)

	data = DataLoader(config)

	# Create placeholders 
	X = tf.placeholder(tf.float32, [None, 32, 32, 1])
	y = tf.placeholder(tf.float32, [None, 10])

	# Create model and logits
	LeNet = model.LeNet(config)
	logits = LeNet.forward(X)

	# Compute metrics
	cost = compute_loss_xent(logits, targets=y)
	accuracy = compute_accuracy(logits, targets=y)

	# Define optimizer 
	optimizer = LeNet.train_optimizer(cost, learning_rate=config["learning_rate"], \
		beta1=0.9, beta2=0.999, epsilon=1e-08)

	# Merging all summaries 
	merged_summary = tf.summary.merge_all()

	# Create saver to save and restore model
	saver = tf.train.Saver(max_to_keep=config["max_to_keep"])

	## Launching the execution graph for training
	with tf.Session() as sess:
		# Initializing all variables
		sess.run(tf.global_variables_initializer())
		# Visualizing the Graph
		writer = tf.summary.FileWriter("./tensorboard/" + config["experiment_name"])
		writer.add_graph(sess.graph)

		for i in range(config["num_epochs"]):
			for j in range(config["num_iter_per_epoch"]):
				# Yield batches of data
				batch_X, batch_y = next(data.next_batch(config["batch_size"]))
				# Run the optimizer
				sess.run(optimizer, feed_dict={X: batch_X, y: batch_y})
				# Compute train loss and accuracy
				loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_X, y: batch_y})

			if (i % config["writer_step"] == 0):
				# Run the merged summary and write it to disk
				s = sess.run(merged_summary, feed_dict={X: batch_X, y: batch_y})
				writer.add_summary(s, (i + 1))

			if (i % config["save_step"] == 0):
				# Saving session
				saver.save(sess, "./saver/" + config["experiment_name"] + "/model_epoch", global_step=(i + 1))

			# Evaluate the validation data
			loss_val, acc_val = sess.run([cost, accuracy], feed_dict={X: data.X_valid, y: data.y_valid})

			if (i % config["display_step"] == 0):
				print("Epoch:", "%03d," % (i + 1), \
					"loss=", "%.5f," % (loss), \
					"train acc=", "%.5f," % (acc), \
					"val loss=", "%.5f," % (loss_val), \
					"val acc=", "%.5f" % (acc_val)
					)

		print("Training complete")


	## Evaluate on test data by loading the saver
	with tf.Session() as sess:
		# Load the network from meta file created by saver
		new_saver = tf.train.import_meta_graph(
				"./saver/" + config["experiment_name"] + "/model_epoch-" + str(config["num_epochs"]) + ".meta")
		# Restore the parameters
		new_saver.restore(sess, tf.train.latest_checkpoint("./saver/" + config["experiment_name"] + "/"))

		loss_test, acc_test = sess.run([cost, accuracy], feed_dict={X: data.X_test, y: data.y_test})

		print("test loss=", "%.5f," % (loss_test), 
			"test accuracy=", "%.5f" % (acc_test))

		print("Testing complete")



if __name__ == '__main__':
	main()