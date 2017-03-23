import sys, configparser, datetime
import baseline, bidirectional, attention
import tensorflow as tf

def prepad(unpadded, pad, size):
	if len(unpadded) == size:
		return unpadded
	return [pad] * (size - len(unpadded)) + unpadded

def feed(model, config, filename):
	batch, length = config.getint('global', 'batch'), config.getint('global', 'steps')
	llist, plist, hlist = list(), list(), list()
	for line in open(filename):
		label, premise, hypothesis = line.split('\t')
		label, premise, hypothesis = int(label), [int(prem) for prem in premise.split()], [int(hypo) for hypo in hypothesis.split()]
		llist.append(label)
		plist.append(prepad(premise, 0, length))
		hlist.append(prepad(hypothesis, 0, length))
		if len(llist) == batch:
			feeddict = dict()
			feeddict.update({model['clabel']: [[1 if i == llist[ii] else 0 for i in xrange(3)] for ii in xrange(batch)]})
			feeddict.update({model['pxi_%i' %i]: [plist[ii][i] for ii in xrange(batch)] for i in xrange(length)})
			feeddict.update({model['hxi_%i' %i]: [hlist[ii][i] for ii in xrange(batch)] for i in xrange(length)})
			yield feeddict
			llist, plist, hlist = list(), list(), list()

def run(model, config, session, summary, filename, train):
	iters, freq, total = config.getint('global', 'iterations') if train else 1, config.getint('global', 'frequency'), 0.
	for i in xrange(iters):
		for ii, feeddict in enumerate(feed(model, config, filename)):
			if train:
				val, t = session.run([model['cce'], model['tce']], feed_dict = feeddict)
				total += val
				if (ii + 1) % freq == 0:
					summ = session.run(model['scce'], feed_dict = feeddict)
					summary.add_summary(summ, model['gsce'].eval())
					print datetime.datetime.now(), i, ii, total
			else:
				val = session.run(model['output'], feed_dict = feeddict)
				total += len(filter(lambda pair: pair[0] == pair[1], zip(np.argmax(val, 1), np.argmax(feeddict[model['clabel']], 1))))
	return total

if __name__ == '__main__':
	config = configparser.ConfigParser(interpolation = configparser.ExtendedInterpolation())
	config.read(sys.argv[1])

	model, modeltype = dict(), config.get('global', 'model')
	model = globals()[modeltype].create(model, config[modeltype])

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		summary = tf.summary.FileWriter(config.get('global', 'logs'), sess.graph)

		print datetime.datetime.now(), run(model, config, sess, summary, sys.argv[2], True)
		print datetime.datetime.now(), run(model, config, sess, summary, sys.argv[3], False)

		tf.train.Saver().save(sess, config.get('global', 'path'))
