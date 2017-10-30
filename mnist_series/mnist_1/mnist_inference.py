# -*- coding: utf-8 -*-
###
# author: houchaoqun
# data: 2017-10-23
# for: 
# 1. definde the process of forward propagation
# 2. definde parameters of the neural network
# 
###
#
# tf.get_variable(name, shape, initializer = tf.constant_initializer(1.0))
# tf.variable_scope(name, reuse)
#
###

import tensorflow as tf

## the parameter of neural network
#
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


def get_weights_variable(shape, regularizer):
	weights = tf.get_variable("weights", shape, initializer = tf.truncated_normal_initializer(stddev = 0.1))
	if regularizer != None:
		tf.add_to_collection("losses", regularizer(weights))
	return weights

def inference(input_tensor, regularizer):
	with tf.variable_scope("layer1"):
		weights = get_weights_variable([INPUT_NODE, LAYER1_NODE], regularizer)
		biases = tf.get_variable("biases", [LAYER1_NODE], initializer = tf.constant_initializer(0.0))
		layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)


	with tf.variable_scope("layer2"):
		weights =  get_weights_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
		biases = tf.get_variable("biases", [OUTPUT_NODE], initializer = tf.constant_initializer(0.0))
		layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)


	return layer2