import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import data_utils as du
import sys
import datetime

def initialize_parameters():
	units_layer1 = int(sys.argv[1])
	units_layer2 = int(sys.argv[2])
	units_layer3 = int(sys.argv[3])

	W1 = tf.Variable(tf.random_normal([784, units_layer1]))
	b1 = tf.Variable(tf.random_normal([units_layer1]))

	W2 = tf.Variable(tf.random_normal([units_layer1, units_layer2]))
	b2 = tf.Variable(tf.random_normal([units_layer2]))

	W3 = tf.Variable(tf.random_normal([units_layer2, units_layer3]))
	b3 = tf.Variable(tf.random_normal([units_layer3]))

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

	Z_out = tf.matmul(A3, W_out) + b_out
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

def nn_train(z_out, y, x, lr, W_out):
	learning_rate = float(sys.argv[4])
	beta = float(sys.argv[5])

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z_out, labels=y)) #Ou reduce_sum
	regularizer = tf.nn.l2_loss(W_out) #L2
	loss = tf.reduce_mean(loss + beta * regularizer)

	optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

	epochs_no = 15
	batch_size = 50
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(epochs_no):
			epoch_loss = 0
			i=0
			while i < len(trainX):
				start = i
				end = i+batch_size
				batch_x = np.array(trainX[start:end])
				batch_y = np.array(trainY[start:end])
				_, c = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y, lr: learning_rate})
				epoch_loss += c
				i+=batch_size
			print('Epoch', epoch+1, 'epoch', epochs_no, 'loss', epoch_loss)
			#if epoch == 10:
				#learning_rate = 0.005 LR after 10 epochs...
		nn_performance_acc(z_out, y)

def nn_performance_acc(z_out, y):
		correct =  tf.equal(tf.argmax(z_out, 1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		acc = accuracy.eval({x:predX, y:predY})
		end = datetime.datetime.now()
		print('***', acc, start, end, os.environ['COMPUTERNAME'], os.path.basename(sys.argv[0]), int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sep="|")

start = datetime.datetime.now()

trainX, trainY, predX, predY, n_classes = du.csv_to_numpy_array("datasets\mnist_train.csv", "datasets\mnist_val.csv")

num_x = trainX.shape[1]
num_y = trainY.shape[1]

x = tf.placeholder(tf.float32, [None, num_x])
y = tf.placeholder(tf.float32, [None, num_y])
lr = tf.placeholder(tf.float32)

parameters = initialize_parameters()
z_out = foward_propagation(x, parameters)
nn_train(z_out, y, x, lr, parameters["W_out"])
