import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/temp/data", one_hot=True)

units_layer1 = 500
units_layer2 = 500
units_layer3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def initialize_parameters():
	W1 = tf.Variable(tf.random_normal([784, units_layer1]))
	b1 = tf.Variable(tf.random_normal([units_layer1]))

	W2 = tf.Variable(tf.random_normal([units_layer1, units_layer2]))
	b2 = tf.Variable(tf.random_normal([units_layer2]))

	W3 = tf.Variable(tf.random_normal([units_layer2, units_layer3]))
	b3 =tf.Variable(tf.random_normal([units_layer3]))

	W_out = tf.Variable(tf.random_normal([units_layer3, n_classes]))
	b_out = tf.Variable(tf.random_normal([n_classes]))

	parameters = {"W1": W1,
				 "b1": b1,
				 "W2": W2,
				 "b2": b2,
				 "W3": W3,
				 "b3": b3,
				 "W_out": W_out,
				 "b_out": b_out}

	return parameters 

def foward_propagation(data, parameters):
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]
	W3 = parameters["W3"]
	b3 = parameters["b3"]
	W_out = parameters["W_out"]
	b_out = parameters["b_out"]

	#Forward Prop
	Z1 = tf.add(tf.matmul(data, W1), b1)
	A1 = tf.nn.relu(Z1)

	Z2 = tf.add(tf.matmul(A1, W2), b2)
	A2 = tf.nn.relu(Z2)

	Z3 = tf.add(tf.matmul(A2, W3), b3)
	A3 = tf.nn.relu(Z3)

	Z_out = tf.add(tf.matmul(A3, W_out), b_out)
	A_out = tf.nn.relu(Z_out)

	cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2,
             "Z3": Z3,
             "A3": A3,
             "Z_out": Z_out,
             "A_out": A_out}

	return Z_out

def nn_train(cache, y, x):

	prediction = cache
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	num_epochs = 5
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(num_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'Num_epoch', num_epochs, 'loss', epoch_loss)

		correct =  tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

parameters = initialize_parameters()
feed_for = foward_propagation(x, parameters)
nn_train(feed_for, y, x)