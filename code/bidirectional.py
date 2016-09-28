import tensorflow as tf

def create(model, config):
	dim_v, dim_i, dim_d, dim_t, dim_b, dim_n, dim_c = config.getint('vocabsize'), config.getint('wvecsize'), config.getint('depth'), config.getint('steps'), config.getint('batch'), config.getint('deepness'), config.getint('classes')
	lrate_ms, dstep_ms, drate_ms, optim_ms = config.getfloat('mslrate'), config.getint('msdstep'), config.getfloat('msdrate'), getattr(tf.train, config.get('msoptim'))
	lrate_ce, dstep_ce, drate_ce, optim_ce = config.getfloat('celrate'), config.getint('cedstep'), config.getfloat('cedrate'), getattr(tf.train, config.get('ceoptim'))

	with tf.name_scope('embedding'):
		model['We'] = tf.Variable(tf.truncated_normal([dim_v, dim_i], stddev = 1.0 / dim_i), name = 'We')
		model['Be'] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'Be')

	with tf.name_scope('plstm'):
		with tf.name_scope('input'):
			for ii in xrange(dim_t):
				model['pxi_%i' %ii] = tf.placeholder(tf.int32, [dim_b], name = 'pxi_%i' %ii)
				model['px_%i' %ii] = tf.add(tf.nn.embedding_lookup(model['We'], model['pxi_%i' %ii]), model['Be'], name = 'px_%i' %ii)

		with tf.name_scope('label'):
			for ii in xrange(dim_t):
				model['pyi_%i' %ii] = tf.placeholder(tf.int32, [dim_b], name = 'pyi_%i' %ii)
				model['py_%i' %ii] = tf.add(tf.nn.embedding_lookup(model['We'], model['pyi_%i' %ii]), model['Be'], name = 'py_%i' %ii)

		for i in xrange(dim_d):
			with tf.name_scope('input_%i' %i):
				for ii in xrange(dim_t):
					model['p+x_%i_%i' %(i, ii)] = model['px_%i' %ii] if i == 0 else model['p+h_%i_%i' %(i - 1, ii)]
					model['p-x_%i_%i' %(i, ii)] = model['px_%i' %ii] if i == 0 else model['p-h_%i_%i' %(i - 1, ii)]

			with tf.name_scope('inputgate_%i' %i):
				model['p+Wi_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'p+Wi_%i' %i)
				model['p+Bi_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'p+Bi_%i' %i)
				model['p-Wi_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'p-Wi_%i' %i)
				model['p-Bi_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'p-Bi_%i' %i)
				for ii in xrange(dim_t):
					model['p+i_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['px_%i_%i' %(i, ii)], model['p+Wi_%i' %i]), model['p+Bi_%i' %i]), name = 'p-i_%i_%i' %(i, ii))
					model['p-i_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['px_%i_%i' %(i, ii)], model['p-Wi_%i' %i]), model['p-Bi_%i' %i]), name = 'p+i_%i_%i' %(i, ii))

			with tf.name_scope('forgetgate_%i' %i):
				model['p+Wf_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'p+Wf_%i' %i)
				model['p+Bf_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'p+Bf_%i' %i)
				model['p-Wf_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'p-Wf_%i' %i)
				model['p-Bf_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'p-Bf_%i' %i)
				for ii in xrange(dim_t):
					model['p+f_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['p+x_%i_%i' %(i, ii)], model['p+Wf_%i' %i]), model['p+Bf_%i' %i]), name = 'p+f_%i_%i' %(i, ii))
					model['p-f_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['p-x_%i_%i' %(i, ii)], model['p-Wf_%i' %i]), model['p-Bf_%i' %i]), name = 'p-f_%i_%i' %(i, ii))

			with tf.name_scope('outputgate_%i' %i):
				model['p+Wo_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'p+Wo_%i' %i)
				model['p+Bo_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'p+Bo_%i' %i)
				model['p-Wo_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'p-Wo_%i' %i)
				model['p-Bo_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'p-Bo_%i' %i)
				for ii in xrange(dim_t):
					model['p+o_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['p+x_%i_%i' %(i, ii)], model['p+Wo_%i' %i]), model['p+Bo_%i' %i]), name = 'p+o_%i_%i' %(i, ii))
					model['p-o_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['p-x_%i_%i' %(i, ii)], model['p-Wo_%i' %i]), model['p-Bo_%i' %i]), name = 'p-o_%i_%i' %(i, ii))

			with tf.name_scope('cellstate_%i' %i):
				model['p+Wc_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'p+Wc_' + str(i))
				model['p+Bc_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'p+Bc_' + str(i))
				model['p-Wc_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'p-Wc_' + str(i))
				model['p-Bc_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'p-Bc_' + str(i))
				for ii in xrange(dim_t):
					model['p+cc_%i_%i' %(i, ii)] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'p+cc_%i_%i' %(i, ii)) if ii == 0 else model['p+c_%i_%i' %(i, ii - 1)] # consider starting with all zeros
					model['p+c_%i_%i' %(i, ii)] = tf.select(tf.equal(model['pxi_%i' %ii], tf.zeros([dim_b], tf.int32)), model['p+cc_%i_%i' %(i, ii)], tf.add(tf.mul(model['p+f_%i_%i' %(i, ii)], model['p+cc_%i_%i' %(i, ii)]), tf.mul(model['p+i_%i_%i' %(i, ii)], tf.nn.tanh(tf.add(tf.matmul(model['px_%i_%i' %(i, ii)], model['p+Wc_%i' %i]), model['p+Bc_%i' %i])))), name = 'p+c_%i_%i' %(i, ii))
				for ii in reversed(xrange(dim_t)):
					model['p-cc_%i_%i' %(i, ii)] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'p-cc_%i_%i' %(i, ii)) if ii == dim_t - 1 else model['p-c_%i_%i' %(i, ii + 1)] # consider starting with all zeros
					model['p-c_%i_%i' %(i, ii)] = tf.select(tf.equal(model['pxi_%i' %ii], tf.zeros([dim_b], tf.int32)), model['p-cc_%i_%i' %(i, ii)], tf.add(tf.mul(model['p-f_%i_%i' %(i, ii)], model['p-cc_%i_%i' %(i, ii)]), tf.mul(model['p-i_%i_%i' %(i, ii)], tf.nn.tanh(tf.add(tf.matmul(model['p-x_%i_%i' %(i, ii)], model['p-Wc_%i' %i]), model['p-Bc_%i' %i])))), name = 'p-c_%i_%i' %(i, ii))

			with tf.name_scope('hidden_%i' %i):
				model['p+Wz_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'p+Wz_%i' %i)
				model['p+Bz_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'p+Bz_%i' %i)
				model['p-Wz_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'p-Wz_%i' %i)
				model['p-Bz_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'p-Bz_%i' %i)
				for ii in xrange(dim_t):
					model['p+z_%i_%i' %(i, ii)] = tf.add(tf.matmul(model['p+c_%i_%i' %(i, ii)], model['p+Wz_%i' %i]), model['p+Bz_%i' %i], name = 'p+z_%i_%i' %(i, ii))
					model['p-z_%i_%i' %(i, ii)] = tf.add(tf.matmul(model['p-c_%i_%i' %(i, ii)], model['p-Wz_%i' %i]), model['p-Bz_%i' %i], name = 'p-z_%i_%i' %(i, ii))

			with tf.name_scope('output_%i' %i):
				for ii in xrange(dim_t):
					model['p+h_%i_%i' %(i, ii)] = tf.mul(model['p+o_%i_%i' %(i, ii)], tf.nn.tanh(model['p+z_%i_%i' %(i, ii)]), name = 'p+h_%i_%i' %(i, ii))
					model['p-h_%i_%i' %(i, ii)] = tf.mul(model['p-o_%i_%i' %(i, ii)], tf.nn.tanh(model['p-z_%i_%i' %(i, ii)]), name = 'p-h_%i_%i' %(i, ii))

		with tf.name_scope('output'):
			for ii in xrange(dim_t):
				model['p+h_%i' %ii] = model['p+h_%i_%i' %(dim_d - 1, ii)]
				model['p-h_%i' %ii] = model['p-h_%i_%i' %(dim_d - 1, ii)]

		with tf.name_scope('meansquared'):
			for ii in xrange(dim_t):
				model['p+ms_%i' %ii] = tf.select(tf.equal(model['pxi_%i' %ii], tf.zeros([dim_b], tf.int32)), tf.zeros([dim_b], tf.float32), tf.reduce_sum(tf.square(tf.sub(model['py_%i' %ii], model['p+h_%i' %ii])), [1]), name = 'p+ms_%i' %ii)
			model['p+ms'] = tf.reduce_sum(tf.add_n([model['p+ms_%i' %ii] for ii in xrange(dim_t)]), name = 'p+ms')
			model['sp+ms'] = tf.scalar_summary(model['p+ms'].name, model['p+ms'])
			for ii in xrange(dim_t):
				model['p-ms_%i' %ii] = tf.select(tf.equal(model['pxi_%i' %ii], tf.zeros([dim_b], tf.int32)), tf.zeros([dim_b], tf.float32), tf.reduce_sum(tf.square(tf.sub(model['py_%i' %ii], model['p-h_%i' %ii])), [1]), name = 'p-ms_%i' %ii)
			model['p-ms'] = tf.reduce_sum(tf.add_n([model['p-ms_%i' %ii] for ii in xrange(dim_t)]), name = 'p-ms')
			model['sp-ms'] = tf.scalar_summary(model['p-ms'].name, model['p-ms'])

	with tf.name_scope('hlstm'):
		with tf.name_scope('input'):
			for ii in xrange(dim_t):
				model['hxi_%i' %ii] = tf.placeholder(tf.int32, [dim_b], name = 'hxi_%i' %ii)
				model['hx_%i' %ii] = tf.add(tf.nn.embedding_lookup(model['We'], model['hxi_%i' %ii]), model['Be'], name = 'hx_%i' %ii)

		with tf.name_scope('label'):
			for ii in xrange(dim_t):
				model['hyi_%i' %ii] = tf.placeholder(tf.int32, [dim_b], name = 'hyi_%i' %ii)
				model['hy_%i' %ii] = tf.add(tf.nn.embedding_lookup(model['We'], model['hyi_%i' %ii]), model['Be'], name = 'hy_%i' %ii)

		for i in xrange(dim_d):
			with tf.name_scope('input_%i' %i):
				for ii in xrange(dim_t):
					model['h+x_%i_%i' %(i, ii)] = model['hx_%i' %ii] if i == 0 else model['h+h_%i_%i' %(i - 1, ii)]
					model['h-x_%i_%i' %(i, ii)] = model['hx_%i' %ii] if i == 0 else model['h-h_%i_%i' %(i - 1, ii)]

			with tf.name_scope('inputgate_%i' %i):
				model['h+Wi_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'h+Wi_%i' %i)
				model['h+Bi_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'h+Bi_%i' %i)
				model['h-Wi_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'h-Wi_%i' %i)
				model['h-Bi_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'h-Bi_%i' %i)
				for ii in xrange(dim_t):
					model['h+i_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['h+x_%i_%i' %(i, ii)], model['h+Wi_%i' %i]), model['h+Bi_%i' %i]), name = 'h+i_%i_%i' %(i, ii))
					model['h-i_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['h-x_%i_%i' %(i, ii)], model['h-Wi_%i' %i]), model['h-Bi_%i' %i]), name = 'h-i_%i_%i' %(i, ii))

			with tf.name_scope('forgetgate_%i' %i):
				model['h+Wf_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'h+Wf_%i' %i)
				model['h+Bf_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'h+Bf_%i' %i)
				model['h-Wf_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'h-Wf_%i' %i)
				model['h-Bf_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'h-Bf_%i' %i)
				for ii in xrange(dim_t):
					model['h+f_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['h+x_%i_%i' %(i, ii)], model['h+Wf_%i' %i]), model['h+Bf_%i' %i]), name = 'h+f_%i_%i' %(i, ii))
					model['h-f_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['h-x_%i_%i' %(i, ii)], model['h-Wf_%i' %i]), model['h-Bf_%i' %i]), name = 'h-f_%i_%i' %(i, ii))

			with tf.name_scope('outputgate_%i' %i):
				model['h+Wo_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'h+Wo_%i' %i)
				model['h+Bo_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'h+Bo_%i' %i)
				model['h-Wo_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'h-Wo_%i' %i)
				model['h-Bo_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'h-Bo_%i' %i)
				for ii in xrange(dim_t):
					model['h+o_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['h+x_%i_%i' %(i, ii)], model['h+Wo_%i' %i]), model['h+Bo_%i' %i]), name = 'h+o_%i_%i' %(i, ii))
					model['h-o_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['h-x_%i_%i' %(i, ii)], model['h-Wo_%i' %i]), model['h-Bo_%i' %i]), name = 'h-o_%i_%i' %(i, ii))

			with tf.name_scope('cellstate_%i' %i):
				model['h+Wc_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'h+Wc_' + str(i))
				model['h+Bc_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'h+Bc_' + str(i))
				model['h-Wc_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'h-Wc_' + str(i))
				model['h-Bc_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'h-Bc_' + str(i))
				for ii in xrange(dim_t):
					model['h+cc_%i_%i' %(i, ii)] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'h+cc_%i_%i' %(i, ii)) if ii == 0 else model['h+c_%i_%i' %(i, ii - 1)] # consider starting with all zeros
					model['h+c_%i_%i' %(i, ii)] = tf.select(tf.equal(model['h+xi_%i' %ii], tf.zeros([dim_b], tf.int32)), model['h+cc_%i_%i' %(i, ii)], tf.add(tf.mul(model['h+f_%i_%i' %(i, ii)], model['h+cc_%i_%i' %(i, ii)]), tf.mul(model['h+i_%i_%i' %(i, ii)], tf.nn.tanh(tf.add(tf.matmul(model['h+x_%i_%i' %(i, ii)], model['h+Wc_%i' %i]), model['h+Bc_%i' %i])))), name = 'h+c_%i_%i' %(i, ii))
				for ii in reversed(xrange(dim_t)):
					model['h-cc_%i_%i' %(i, ii)] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'h-cc_%i_%i' %(i, ii)) if ii == dim_t - 1 else model['h-c_%i_%i' %(i, ii + 1)] # consider starting with all zeros
					model['h-c_%i_%i' %(i, ii)] = tf.select(tf.equal(model['h-xi_%i' %ii], tf.zeros([dim_b], tf.int32)), model['h-cc_%i_%i' %(i, ii)], tf.add(tf.mul(model['h-f_%i_%i' %(i, ii)], model['h-cc_%i_%i' %(i, ii)]), tf.mul(model['h-i_%i_%i' %(i, ii)], tf.nn.tanh(tf.add(tf.matmul(model['h-x_%i_%i' %(i, ii)], model['h-Wc_%i' %i]), model['h-Bc_%i' %i])))), name = 'h-c_%i_%i' %(i, ii))

			with tf.name_scope('hidden_%i' %i):
				model['h+Wz_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'h+Wz_%i' %i)
				model['h+Bz_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'h+Bz_%i' %i)
				model['h-Wz_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'h-Wz_%i' %i)
				model['h-Bz_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'h-Bz_%i' %i)
				for ii in xrange(dim_t):
					model['h+z_%i_%i' %(i, ii)] = tf.add(tf.matmul(model['h+c_%i_%i' %(i, ii)], model['h+Wz_%i' %i]), model['h+Bz_%i' %i], name = 'h+z_%i_%i' %(i, ii))
					model['h-z_%i_%i' %(i, ii)] = tf.add(tf.matmul(model['h-c_%i_%i' %(i, ii)], model['h-Wz_%i' %i]), model['h-Bz_%i' %i], name = 'h-z_%i_%i' %(i, ii))

			with tf.name_scope('output_%i' %i):
				for ii in xrange(dim_t):
					model['h+h_%i_%i' %(i, ii)] = tf.mul(model['h+o_%i_%i' %(i, ii)], tf.nn.tanh(model['h+z_%i_%i' %(i, ii)]), name = 'h+h_%i_%i' %(i, ii))
					model['h-h_%i_%i' %(i, ii)] = tf.mul(model['h-o_%i_%i' %(i, ii)], tf.nn.tanh(model['h-z_%i_%i' %(i, ii)]), name = 'h-h_%i_%i' %(i, ii))

		with tf.name_scope('output'):
			for ii in xrange(dim_t):
				model['h+h_%i' %ii] = model['h+h_%i_%i' %(dim_d - 1, ii)]
				model['h-h_%i' %ii] = model['h-h_%i_%i' %(dim_d - 1, ii)]

		with tf.name_scope('meansquared'):
			for ii in xrange(dim_t):
				model['h+ms_%i' %ii] = tf.select(tf.equal(model['hxi_%i' %ii], tf.zeros([dim_b], tf.int32)), tf.zeros([dim_b], tf.float32), tf.reduce_sum(tf.square(tf.sub(model['hy_%i' %ii], model['h+h_%i' %ii])), [1]), name = 'h+ms_%i' %ii)
			model['h+ms'] = tf.reduce_sum(tf.add_n([model['h+ms_%i' %ii] for ii in xrange(dim_t)]), name = 'h+ms')
			model['sh+ms'] = tf.scalar_summary(model['h+ms'].name, model['h+ms'])
			for ii in xrange(dim_t):
				model['h-ms_%i' %ii] = tf.select(tf.equal(model['hxi_%i' %ii], tf.zeros([dim_b], tf.int32)), tf.zeros([dim_b], tf.float32), tf.reduce_sum(tf.square(tf.sub(model['hy_%i' %ii], model['h-h_%i' %ii])), [1]), name = 'h-ms_%i' %ii)
			model['h-ms'] = tf.reduce_sum(tf.add_n([model['h-ms_%i' %ii] for ii in xrange(dim_t)]), name = 'h-ms')
			model['sh-ms'] = tf.scalar_summary(model['h-ms'].name, model['h-ms'])

	with tf.name_scope('classification'):
		with tf.name_scope('label'):
			model['clabel'] = tf.placeholder(tf.float32, [dim_b, dim_c], name = 'clabel')

		for i in xrange(dim_n):
			with tf.name_scope('layer_%i' %i):
				model['cW_%i' %i] = tf.Variable(tf.truncated_normal([4 * dim_i, 4 * dim_i], stddev = 0.25 / dim_i), name = 'cW_%i' %i) if i != dim_n - 1 else tf.Variable(tf.truncated_normal([4 * dim_i, dim_c], stddev = 1.0 / dim_c), name = 'cW_%i' %i)
				model['cB_%i' %i] = tf.Variable(tf.truncated_normal([1, 4 * dim_i], stddev = 0.25 / dim_i), name = 'cB_%i' %i) if i != dim_n - 1 else tf.Variable(tf.truncated_normal([1, dim_c], stddev = 1.0 / dim_c), name = 'cB_%i' %i)
				model['cx_%i' %i] = tf.concat(1, [model['p+h_%i' %(dim_t - 1)], [model['p-h_%i' %(0)], model['h+h_%i' %(dim_t - 1)], model['h-h_%i' %(0)]], name = 'cx_%i' %i) if i == 0 else model['cy_%i' %(i - 1)]
				model['cy_%i' %i] = tf.add(tf.matmul(model['cx_%i' %i], model['cW_%i' %i]), model['cB_%i' %i], name = 'cy_%i' %i)

		with tf.name_scope('output'):
			model['output'] = tf.nn.softmax(model['cy_%i' %(dim_n - 1)], name = 'output')

		with tf.name_scope('crossentropy'):
			model['cce'] = tf.reduce_sum(-tf.mul(model['clabel'], tf.log(model['output'])), name = 'cce')

	model['gsms'] = tf.Variable(0, trainable = False, name = 'gsms')
	model['lrms'] = tf.train.exponential_decay(lrate_ms, model['gsms'], dstep_ms, drate_ms, staircase = False, name = 'lrms')
	model['tms'] = optim_ms(model['lrms']).minimize(model['p+ms'] + model['p-ms'] + model['h+ms'] + model['h-ms'], global_step = model['gsms'], name = 'tms')

	model['gsce'] = tf.Variable(0, trainable = False, name = 'gsce')
	model['lrce'] = tf.train.exponential_decay(lrate_ce, model['gsce'], dstep_ce, drate_ce, staircase = False, name = 'lrce')
	model['tce'] = optim_ce(model['lrce']).minimize(model['cce'], global_step = model['gsce'], name = 'tce')

	return model
