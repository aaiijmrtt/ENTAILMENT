import sys, configparser
import baseline, bidirectional, attention
import tensorflow as tf

if __name__ == '__main__':
	config = configparser.ConfigParser(interpolation = configparser.ExtendedInterpolation())
	config.read(sys.argv[1])

	model, modeltype = dict(), config.get('global', 'model')
	model = globals()[modeltype].create(model, config[modeltype])

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		tf.train.SummaryWriter(config.get('global', 'logs'), sess.graph)
		saver.save(sess, config.get('global', 'path'))
