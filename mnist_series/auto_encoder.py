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
class AdditiveGaussianNoiseAutoencoder(object):
	def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer(), scale = 0.1):
		self.n_input = n_input
		self.n_hidden = n_hidden	# 目前只是用了一个隐含层，可根据实际情况增加隐含层数量
		self.transfer = transfer_function	# 隐含层激活函数
		self.scale = tf.placeholder(tf.float32)
		self.training_scale = scale 	# 高斯噪声系数

		# 初始化网络权重
		network_weights = self._initialize_weights()
		self.weights = network_weights

		# 定义网络结构
		self.x = tf.placeholder(tf.float32, [None, self.n_input])
		self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']), self.weights['b1']))	# 
		self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])	# tf.add() 和 “直接使用 + 操作”的区别，还有tf.add_n()函数

		# 定义自编码的损失函数
		self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
		self.optimizer = optimizer.minimize(self.cost)

		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)

	def _initialize_weights(self):
		all_weights = dict()
		# n_input = n_output

		all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
		all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], tf.float32))
		all_weights['w2'] = tf.Variable(xavier_init(self.n_hidden, self.n_input))
		all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], tf.float32))

		return all_weights


	### 
	# 作用：用一个 batch 数据进行训练，并返回当前的损失 cost
	# 
	def partial_fit(self, X):
		cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict = {self.x: X, self.scale: self.training_scale})
		return cost

	###
	# 作用：只求损失函数的值；
	# 什么时候使用：在自编码器训练完毕后，在测试集上对模型进行评测时会用到。
	def calc_total_cost(self, X):
		return self.sess.run(self.cost, feed_dict = {self.x: X, self.scale: self.training_scale})

	### 
	# 作用：返回自编码器隐含层的输出结果，提供一个接口来获取抽象后的特征
	# 自编码器的隐含层：最主要的功能就是学习出数据中的高阶特征
	def transform(self, X):
		return self.sess.run(self.hidden, feed_dict = {self.x: X, self.scale: training_scale})

	### 
	# 作用：将隐含层的输出结果作为输入，通过之后的重建层将提取得到的高阶特征复原为原始数据
	# 
	def generate(self, hidden = None):
		if hidden is None:
			hidden = np.random_normal(size = self.weights['b1'])
		return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})

	### 
	# 作用：整体运行一遍复原过程，包括提取高阶特征、通过高阶特征复原数据
	# 输入为原始数据，输出为复员后的数据
	# 
	def reconstruct(self, X):
		return self.sess.run(self.reconstruction, feed_dict = {self.x: X, self.scale: self.training_scale})


	def getWeights(self):
		return self.sess.run(self.weights['w1'])

	def getBiases(self):
		return self.sess.run(self.weights['b1'])


### 
# 作用：对训练数据、测试数据进行标准化
# - 让数据变成 0 均值：减去均值
# - 让数据的标准差为1：除以标准差
# 注意：必须保证训练、测试数据都使用完全相同的Scale，才能保证后面模块处理数据的一致性。
# 
def standard_scale(X_train, X_test):
	preprocessor = prep.StandardScaler().fit(X_train)
	X_train = preprocessor.transform(X_train)
	X_test = preprocessor.transform(X_test)

	return X_train, X_test

### 
# 作用：随机获取block数据区域：取一个从0到len(data) - batch_size之间的随机整数start_index，以start_index为起始位置，按顺序获取一个长度为batch_size的数据区域
# 属于不放回抽样，提高数据的利用效率。
# 
def get_random_block_from_data(data, batch_size):
	start_index = np.random.randint(0, len(data) - batch_size)
	return data[start_index:(start_index + batch_size)]

### 
# 
# 
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)	# 对数据集和测试集进行标准化处理
n_samples = int(mnist.train.num_examples)
training_epochs = 2000
batch_size = 128
display_step = 1

autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784,
										n_hidden = 200,
										transfer_function = tf.nn.softplus,
										optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
										scale = 0.1
	)


total_batch = int(n_samples / batch_size)
for epoch in range(training_epochs):
	avg_cost = 0.
	for i in range(total_batch):
		batch_xs = get_random_block_from_data(X_train, batch_size)
		cost = autoencoder.partial_fit(batch_xs)
		avg_cost += cost/n_samples * batch_size
		
	if epoch % display_step == 0:
		print("Epoch:", '%04d' % (epoch + 1), "cost = ", "{:.9f}".format(avg_cost))

print("n_samples = %d" % n_samples)
print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))




