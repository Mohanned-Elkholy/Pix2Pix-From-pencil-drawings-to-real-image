import tensorflow as tf

def get_xtrain():
	""" This function gets the training data """
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
	return x_train
