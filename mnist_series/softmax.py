# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10
TRAINING_STEPS = 20000
BATCH_SIZE = 100

def dataset_info(mnist):
	print(mnist.train.images.shape, mnist.train.labels.shape)
	print(mnist.test.images.shape, mnist.test.labels.shape)
	print(mnist.validation.images.shape, mnist.validation.labels.shape)
	print("...")


def softmax_model(mnist):
	x_input = tf.placeholder(tf.float32, [None, INPUT_NODE], "x_input")
	y_labels = tf.placeholder(tf.float32, [None, OUTPUT_NODE], "y_labels")
	weights = tf.get_variable("weights", [INPUT_NODE, OUTPUT_NODE], initializer = tf.truncated_normal_initializer(stddev = 0.1))
	biases = tf.get_variable("biases", [OUTPUT_NODE], initializer = tf.constant_initializer(0.0))

	# y_result = tf.nn.relu(tf.matmul(x_input, weights) + biases)
	y_result = tf.nn.softmax(tf.matmul(x_input, weights) + biases)
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_labels * tf.log(y_result), reduction_indices = [1]))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		for i in range(TRAINING_STEPS):
			batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
			# run it !!!!!
			sess.run(train_step, feed_dict = {x_input: batch_xs, y_labels: batch_ys})

			if (i%500) == 0:
				v_correct_prediction = tf.equal(tf.argmax(y_labels, 1), tf.argmax(y_result, 1))
				v_accuracy = tf.reduce_mean(tf.cast(v_correct_prediction, tf.float32))
				batch_validation_xs, batch_validation_ys = mnist.validation.next_batch(BATCH_SIZE)
				acc_validation = sess.run(v_accuracy, feed_dict = {x_input: batch_validation_xs, y_labels: batch_validation_ys})
				print("After %d training step(s), acc = %g" % (i, acc_validation))

		# set test image batch.
		# Before this, the model has been trained.
		acc = 0
		num_test_image = mnist.test.images.shape[0]
		accounts = num_test_image/BATCH_SIZE
		correct_prediction = tf.equal(tf.argmax(y_labels, 1), tf.argmax(y_result, 1))
		for it in range(accounts):
			# correct_prediction = tf.equal(tf.argmax(y_labels, 1), tf.argmax(y_result, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			#acc += accuracy
			batch_test_xs, batch_test_ys = mnist.test.next_batch(BATCH_SIZE)
			acc += sess.run(accuracy, feed_dict = {x_input: batch_test_xs, y_labels: batch_test_ys})
		print("acc = %g"% (acc/accounts))


def main(argv = None):
	mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
	dataset_info(mnist)
	softmax_model(mnist)

if __name__ == '__main__':
	tf.app.run()
