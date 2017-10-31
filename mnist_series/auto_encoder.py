# -*- coding: utf-8 -*-

### 自编码器 - 理论知识 ###
# - 无监督，一种神经网络，使用自身的高阶特征编码自己。
# - 很难直接训练极深的网络，但可以用无监督的逐层提取特征，将网络的权重初始化到一个比较好的位置，辅助后续的监督训练。
# - 1）输入和输出一致；2）借助稀疏编码的思想，使用稀疏的一些高阶特征重新组合来重构自己（不是简单的复制）
# - 限制：1）隐层节点数小于输入/输出的节点数；2）给数据加入噪声（如，加性高斯噪声）
# - 自编码器的作用：1）为监督学习做预训练；2）特征提取+分析，现实生活中未标注的数据居多，自编码器很有发展前景。
#
### 基于 TensorFlow 实现自编码器 
# - 1）无噪声的自编码器
# - 2）高斯去噪自编码器
# - 3）Masking Noise的自编码器
# - 4）Variational AutoEncoder（VAE）
# 
# 一些所需的函数库
# - xavier initialization：参数初始化器，根据某一层网络的输入、输出节点数量自动调整最合适的分布（让权重被初始化得正合适）
# 
###

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Xavier 初始化器满足：
# 1）0均值
# 2）方差为 2/(n_in + n_out)
# 3）分布：均匀分布 or 高斯分布
def xavier_init(fan_in, fan_out, constant = 1):
	# fan_in: 输入节点的数量
	# fan_out: 输出节点的数量
	low = -constant * np.sqrt(6.0/(fan_in + fan_out))
	high = constant * np.sqrt(6.0/(fan_in + fan_out))

	return tf.random_uniform((fan_in, fan_out), minval = low, maxval = high, dtype = tf.float32)


# 去噪自编码的 class
# 
class AdditiveGaussianNoiseAutoencoder(object):
	def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer(), scale = 0.1):
		self.n_input = n_input
		self.n_hidden = n_hidden	# 目前只是用了一个隐含层，可根据实际情况增加隐含层数量
		self.transfer = transfer_function	# 隐含层激活函数
		self.scale = tf.placeholder(tf.float32)
		self.training_scale = scale 	# 高斯噪声系数

		# 初始化网络权重
		network_weights = self.__initialize_weights()
		self.weights = network_weights

		# 定义网络结构
		self.x = tf.placeholder(tf.float32, [None, self.n_input])
		self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal(n_input,)), self.weights['w1'], self.weights['b1']))	# 
		self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

		# 定义自编码的损失函数
		self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
		self.optimizer = optimizer.minimize(self.cost)

		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)


