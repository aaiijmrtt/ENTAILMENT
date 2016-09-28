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
					model['pFx_%i_%i' %(i, ii)] = model['px_%i' %ii] if i == 0 else model['pFh_%i_%i' %(i - 1, ii)]
					model['pBx_%i_%i' %(i, ii)] = model['px_%i' %ii] if i == 0 else model['pBh_%i_%i' %(i - 1, ii)]

			with tf.name_scope('inputgate_%i' %i):
				model['pFWi_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pFWi_%i' %i)
				model['pFBi_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pFBi_%i' %i)
				model['pBWi_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pBWi_%i' %i)
				model['pBBi_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBBi_%i' %i)
				for ii in xrange(dim_t):
					model['pFi_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['pFx_%i_%i' %(i, ii)], model['pFWi_%i' %i]), model['pFBi_%i' %i]), name = 'pFi_%i_%i' %(i, ii))
					model['pBi_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['pBx_%i_%i' %(i, ii)], model['pBWi_%i' %i]), model['pBBi_%i' %i]), name = 'pBi_%i_%i' %(i, ii))

			with tf.name_scope('forgetgate_%i' %i):
				model['pFWf_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pFWf_%i' %i)
				model['pFBf_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pFBf_%i' %i)
				model['pBWf_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pBWf_%i' %i)
				model['pBBf_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBBf_%i' %i)
				for ii in xrange(dim_t):
					model['pFf_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['pFx_%i_%i' %(i, ii)], model['pFWf_%i' %i]), model['pFBf_%i' %i]), name = 'pFf_%i_%i' %(i, ii))
					model['pBf_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['pBx_%i_%i' %(i, ii)], model['pBWf_%i' %i]), model['pBBf_%i' %i]), name = 'pBf_%i_%i' %(i, ii))

			with tf.name_scope('outputgate_%i' %i):
				model['pFWo_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pFWo_%i' %i)
				model['pFBo_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pFBo_%i' %i)
				model['pBWo_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pBWo_%i' %i)
				model['pBBo_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBBo_%i' %i)
				for ii in xrange(dim_t):
					model['pFo_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['pFx_%i_%i' %(i, ii)], model['pFWo_%i' %i]), model['pFBo_%i' %i]), name = 'pFo_%i_%i' %(i, ii))
					model['pBo_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['pBx_%i_%i' %(i, ii)], model['pBWo_%i' %i]), model['pBBo_%i' %i]), name = 'pBo_%i_%i' %(i, ii))

			with tf.name_scope('cellstate_%i' %i):
				model['pFWc_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pFWc_' + str(i))
				model['pFBc_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pFBc_' + str(i))
				model['pBWc_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pBWc_' + str(i))
				model['pBBc_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBBc_' + str(i))
				for ii in xrange(dim_t):
					model['pFcc_%i_%i' %(i, ii)] = tf.Variable(tf.truncated_normal([dim_b, dim_i], stddev = 1.0 / dim_i), name = 'pFcc_%i_%i' %(i, ii)) if ii == 0 else model['pFc_%i_%i' %(i, ii - 1)] # consider starting with all zeros
					model['pFc_%i_%i' %(i, ii)] = tf.select(tf.equal(model['pxi_%i' %ii], tf.zeros([dim_b], tf.int32)), model['pFcc_%i_%i' %(i, ii)], tf.add(tf.mul(model['pFf_%i_%i' %(i, ii)], model['pFcc_%i_%i' %(i, ii)]), tf.mul(model['pFi_%i_%i' %(i, ii)], tf.nn.tanh(tf.add(tf.matmul(model['pFx_%i_%i' %(i, ii)], model['pFWc_%i' %i]), model['pFBc_%i' %i])))), name = 'pFc_%i_%i' %(i, ii))
				for ii in reversed(xrange(dim_t)):
					model['pBcc_%i_%i' %(i, ii)] = tf.Variable(tf.truncated_normal([dim_b, dim_i], stddev = 1.0 / dim_i), name = 'pBcc_%i_%i' %(i, ii)) if ii == dim_t - 1 else model['pBc_%i_%i' %(i, ii + 1)] # consider starting with all zeros
					model['pBc_%i_%i' %(i, ii)] = tf.select(tf.equal(model['pxi_%i' %ii], tf.zeros([dim_b], tf.int32)), model['pBcc_%i_%i' %(i, ii)], tf.add(tf.mul(model['pBf_%i_%i' %(i, ii)], model['pBcc_%i_%i' %(i, ii)]), tf.mul(model['pBi_%i_%i' %(i, ii)], tf.nn.tanh(tf.add(tf.matmul(model['pBx_%i_%i' %(i, ii)], model['pBWc_%i' %i]), model['pBBc_%i' %i])))), name = 'pBc_%i_%i' %(i, ii))

			with tf.name_scope('hidden_%i' %i):
				model['pFWz_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pFWz_%i' %i)
				model['pFBz_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pFBz_%i' %i)
				model['pBWz_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pBWz_%i' %i)
				model['pBBz_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBBz_%i' %i)
				for ii in xrange(dim_t):
					model['pFz_%i_%i' %(i, ii)] = tf.add(tf.matmul(model['pFc_%i_%i' %(i, ii)], model['pFWz_%i' %i]), model['pFBz_%i' %i], name = 'pFz_%i_%i' %(i, ii))
					model['pBz_%i_%i' %(i, ii)] = tf.add(tf.matmul(model['pBc_%i_%i' %(i, ii)], model['pBWz_%i' %i]), model['pBBz_%i' %i], name = 'pBz_%i_%i' %(i, ii))

			with tf.name_scope('output_%i' %i):
				for ii in xrange(dim_t):
					model['pFh_%i_%i' %(i, ii)] = tf.mul(model['pFo_%i_%i' %(i, ii)], tf.nn.tanh(model['pFz_%i_%i' %(i, ii)]), name = 'pFh_%i_%i' %(i, ii))
					model['pBh_%i_%i' %(i, ii)] = tf.mul(model['pBo_%i_%i' %(i, ii)], tf.nn.tanh(model['pBz_%i_%i' %(i, ii)]), name = 'pBh_%i_%i' %(i, ii))
				model['pFh_%i_%i' %(dim_d - 1, -1)] = tf.zeros([dim_b, dim_i], tf.float32)
				model['pBh_%i_%i' %(dim_d - 1, dim_t)] = tf.zeros([dim_b, dim_i], tf.float32)

		with tf.name_scope('output'):
			for ii in xrange(dim_t):
				model['pFh_%i' %ii] = tf.select(tf.equal(model['pxi_%i' %ii], tf.zeros([dim_b], tf.int32)), model['pFh_%i_%i' %(dim_d - 1, ii - 1)], model['pFh_%i_%i' %(dim_d - 1, ii)], name = 'pFh_%i' %ii)
				model['pBh_%i' %ii] = tf.select(tf.equal(model['pxi_%i' %ii], tf.zeros([dim_b], tf.int32)), model['pBh_%i_%i' %(dim_d - 1, ii + 1)], model['pBh_%i_%i' %(dim_d - 1, ii)], name = 'pBh_%i' %ii)

		with tf.name_scope('meansquared'):
			for ii in xrange(dim_t):
				model['pFms_%i' %ii] = tf.select(tf.equal(model['pxi_%i' %ii], tf.zeros([dim_b], tf.int32)), tf.zeros([dim_b], tf.float32), tf.reduce_sum(tf.square(tf.sub(model['py_%i' %ii], model['pFh_%i' %ii])), [1]), name = 'pFms_%i' %ii)
			model['pFms'] = tf.reduce_sum(tf.add_n([model['pFms_%i' %ii] for ii in xrange(dim_t)]), name = 'pFms')
			model['sp+ms'] = tf.scalar_summary(model['pFms'].name, model['pFms'])
			for ii in xrange(dim_t):
				model['pBms_%i' %ii] = tf.select(tf.equal(model['pxi_%i' %ii], tf.zeros([dim_b], tf.int32)), tf.zeros([dim_b], tf.float32), tf.reduce_sum(tf.square(tf.sub(model['py_%i' %ii], model['pBh_%i' %ii])), [1]), name = 'pBms_%i' %ii)
			model['pBms'] = tf.reduce_sum(tf.add_n([model['pBms_%i' %ii] for ii in xrange(dim_t)]), name = 'pBms')
			model['sp-ms'] = tf.scalar_summary(model['pBms'].name, model['pBms'])

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
					model['hFx_%i_%i' %(i, ii)] = model['hx_%i' %ii] if i == 0 else model['hFh_%i_%i' %(i - 1, ii)]
					model['hBx_%i_%i' %(i, ii)] = model['hx_%i' %ii] if i == 0 else model['hBh_%i_%i' %(i - 1, ii)]

			with tf.name_scope('inputgate_%i' %i):
				model['hFWi_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'hFWi_%i' %i)
				model['hFBi_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'hFBi_%i' %i)
				model['hBWi_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'hBWi_%i' %i)
				model['hBBi_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'hBBi_%i' %i)
				for ii in xrange(dim_t):
					model['hFi_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['hFx_%i_%i' %(i, ii)], model['hFWi_%i' %i]), model['hFBi_%i' %i]), name = 'hFi_%i_%i' %(i, ii))
					model['hBi_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['hBx_%i_%i' %(i, ii)], model['hBWi_%i' %i]), model['hBBi_%i' %i]), name = 'hBi_%i_%i' %(i, ii))

			with tf.name_scope('forgetgate_%i' %i):
				model['hFWf_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'hFWf_%i' %i)
				model['hFBf_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'hFBf_%i' %i)
				model['hBWf_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'hBWf_%i' %i)
				model['hBBf_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'hBBf_%i' %i)
				for ii in xrange(dim_t):
					model['hFf_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['hFx_%i_%i' %(i, ii)], model['hFWf_%i' %i]), model['hFBf_%i' %i]), name = 'hFf_%i_%i' %(i, ii))
					model['hBf_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['hBx_%i_%i' %(i, ii)], model['hBWf_%i' %i]), model['hBBf_%i' %i]), name = 'hBf_%i_%i' %(i, ii))

			with tf.name_scope('outputgate_%i' %i):
				model['hFWo_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'hFWo_%i' %i)
				model['hFBo_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'hFBo_%i' %i)
				model['hBWo_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'hBWo_%i' %i)
				model['hBBo_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'hBBo_%i' %i)
				for ii in xrange(dim_t):
					model['hFo_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['hFx_%i_%i' %(i, ii)], model['hFWo_%i' %i]), model['hFBo_%i' %i]), name = 'hFo_%i_%i' %(i, ii))
					model['hBo_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['hBx_%i_%i' %(i, ii)], model['hBWo_%i' %i]), model['hBBo_%i' %i]), name = 'hBo_%i_%i' %(i, ii))

			with tf.name_scope('cellstate_%i' %i):
				model['hFWc_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'hFWc_' + str(i))
				model['hFBc_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'hFBc_' + str(i))
				model['hBWc_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'hBWc_' + str(i))
				model['hBBc_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'hBBc_' + str(i))
				for ii in xrange(dim_t):
					model['hFcc_%i_%i' %(i, ii)] = tf.Variable(tf.truncated_normal([dim_b, dim_i], stddev = 1.0 / dim_i), name = 'hFcc_%i_%i' %(i, ii)) if ii == 0 else model['hFc_%i_%i' %(i, ii - 1)] # consider starting with all zeros
					model['hFc_%i_%i' %(i, ii)] = tf.select(tf.equal(model['hxi_%i' %ii], tf.zeros([dim_b], tf.int32)), model['hFcc_%i_%i' %(i, ii)], tf.add(tf.mul(model['hFf_%i_%i' %(i, ii)], model['hFcc_%i_%i' %(i, ii)]), tf.mul(model['hFi_%i_%i' %(i, ii)], tf.nn.tanh(tf.add(tf.matmul(model['hFx_%i_%i' %(i, ii)], model['hFWc_%i' %i]), model['hFBc_%i' %i])))), name = 'hFc_%i_%i' %(i, ii))
				for ii in reversed(xrange(dim_t)):
					model['hBcc_%i_%i' %(i, ii)] = tf.Variable(tf.truncated_normal([dim_b, dim_i], stddev = 1.0 / dim_i), name = 'hBcc_%i_%i' %(i, ii)) if ii == dim_t - 1 else model['hBc_%i_%i' %(i, ii + 1)] # consider starting with all zeros
					model['hBc_%i_%i' %(i, ii)] = tf.select(tf.equal(model['hxi_%i' %ii], tf.zeros([dim_b], tf.int32)), model['hBcc_%i_%i' %(i, ii)], tf.add(tf.mul(model['hBf_%i_%i' %(i, ii)], model['hBcc_%i_%i' %(i, ii)]), tf.mul(model['hBi_%i_%i' %(i, ii)], tf.nn.tanh(tf.add(tf.matmul(model['hBx_%i_%i' %(i, ii)], model['hBWc_%i' %i]), model['hBBc_%i' %i])))), name = 'hBc_%i_%i' %(i, ii))

			with tf.name_scope('hidden_%i' %i):
				model['hFWz_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'hFWz_%i' %i)
				model['hFBz_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'hFBz_%i' %i)
				model['hBWz_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'hBWz_%i' %i)
				model['hBBz_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'hBBz_%i' %i)
				for ii in xrange(dim_t):
					model['hFz_%i_%i' %(i, ii)] = tf.add(tf.matmul(model['hFc_%i_%i' %(i, ii)], model['hFWz_%i' %i]), model['hFBz_%i' %i], name = 'hFz_%i_%i' %(i, ii))
					model['hBz_%i_%i' %(i, ii)] = tf.add(tf.matmul(model['hBc_%i_%i' %(i, ii)], model['hBWz_%i' %i]), model['hBBz_%i' %i], name = 'hBz_%i_%i' %(i, ii))

			with tf.name_scope('output_%i' %i):
				for ii in xrange(dim_t):
					model['hFh_%i_%i' %(i, ii)] = tf.mul(model['hFo_%i_%i' %(i, ii)], tf.nn.tanh(model['hFz_%i_%i' %(i, ii)]), name = 'hFh_%i_%i' %(i, ii))
					model['hBh_%i_%i' %(i, ii)] = tf.mul(model['hBo_%i_%i' %(i, ii)], tf.nn.tanh(model['hBz_%i_%i' %(i, ii)]), name = 'hBh_%i_%i' %(i, ii))
				model['hFh_%i_%i' %(dim_d - 1, -1)] = tf.zeros([dim_b, dim_i], tf.float32)
				model['hBh_%i_%i' %(dim_d - 1, dim_t)] = tf.zeros([dim_b, dim_i], tf.float32)

		with tf.name_scope('output'):
			for ii in xrange(dim_t):
				model['hFh_%i' %ii] = tf.select(tf.equal(model['hxi_%i' %ii], tf.zeros([dim_b], tf.int32)), model['hFh_%i_%i' %(dim_d - 1, ii - 1)], model['hFh_%i_%i' %(dim_d - 1, ii)], name = 'hFh_%i' %ii)
				model['hBh_%i' %ii] = tf.select(tf.equal(model['hxi_%i' %ii], tf.zeros([dim_b], tf.int32)), model['hBh_%i_%i' %(dim_d - 1, ii + 1)], model['hBh_%i_%i' %(dim_d - 1, ii)], name = 'hBh_%i' %ii)

		with tf.name_scope('meansquared'):
			for ii in xrange(dim_t):
				model['hFms_%i' %ii] = tf.select(tf.equal(model['hxi_%i' %ii], tf.zeros([dim_b], tf.int32)), tf.zeros([dim_b], tf.float32), tf.reduce_sum(tf.square(tf.sub(model['hy_%i' %ii], model['hFh_%i' %ii])), [1]), name = 'hFms_%i' %ii)
			model['hFms'] = tf.reduce_sum(tf.add_n([model['hFms_%i' %ii] for ii in xrange(dim_t)]), name = 'hFms')
			model['sh+ms'] = tf.scalar_summary(model['hFms'].name, model['hFms'])
			for ii in xrange(dim_t):
				model['hBms_%i' %ii] = tf.select(tf.equal(model['hxi_%i' %ii], tf.zeros([dim_b], tf.int32)), tf.zeros([dim_b], tf.float32), tf.reduce_sum(tf.square(tf.sub(model['hy_%i' %ii], model['hBh_%i' %ii])), [1]), name = 'hBms_%i' %ii)
			model['hBms'] = tf.reduce_sum(tf.add_n([model['hBms_%i' %ii] for ii in xrange(dim_t)]), name = 'hBms')
			model['sh-ms'] = tf.scalar_summary(model['hBms'].name, model['hBms'])

	with tf.name_scope('classification'):
		with tf.name_scope('label'):
			model['clabel'] = tf.placeholder(tf.float32, [dim_b, dim_c], name = 'clabel')

		for i in xrange(dim_n):
			with tf.name_scope('layer_%i' %i):
				model['cW_%i' %i] = tf.Variable(tf.truncated_normal([4 * dim_i, 4 * dim_i], stddev = 0.25 / dim_i), name = 'cW_%i' %i) if i != dim_n - 1 else tf.Variable(tf.truncated_normal([4 * dim_i, dim_c], stddev = 1.0 / dim_c), name = 'cW_%i' %i)
				model['cB_%i' %i] = tf.Variable(tf.truncated_normal([1, 4 * dim_i], stddev = 0.25 / dim_i), name = 'cB_%i' %i) if i != dim_n - 1 else tf.Variable(tf.truncated_normal([1, dim_c], stddev = 1.0 / dim_c), name = 'cB_%i' %i)
				model['cx_%i' %i] = tf.concat(1, [model['pFh_%i' %(dim_t - 1)], model['pBh_%i' %(0)], model['hFh_%i' %(dim_t - 1)], model['hBh_%i' %(0)]], name = 'cx_%i' %i) if i == 0 else model['cy_%i' %(i - 1)]
				model['cy_%i' %i] = tf.add(tf.matmul(model['cx_%i' %i], model['cW_%i' %i]), model['cB_%i' %i], name = 'cy_%i' %i)

		with tf.name_scope('output'):
			model['output'] = tf.nn.softmax(model['cy_%i' %(dim_n - 1)], name = 'output')

		with tf.name_scope('crossentropy'):
			model['cce'] = tf.reduce_sum(-tf.mul(model['clabel'], tf.log(model['output'])), name = 'cce')
			model['scce'] = tf.scalar_summary(model['cce'].name, model['cce'])

	model['gsms'] = tf.Variable(0, trainable = False, name = 'gsms')
	model['lrms'] = tf.train.exponential_decay(lrate_ms, model['gsms'], dstep_ms, drate_ms, staircase = False, name = 'lrms')
	model['tms'] = optim_ms(model['lrms']).minimize(model['pFms'] + model['pBms'] + model['hFms'] + model['hBms'], global_step = model['gsms'], name = 'tms')

	model['gsce'] = tf.Variable(0, trainable = False, name = 'gsce')
	model['lrce'] = tf.train.exponential_decay(lrate_ce, model['gsce'], dstep_ce, drate_ce, staircase = False, name = 'lrce')
	model['tce'] = optim_ce(model['lrce']).minimize(model['cce'], global_step = model['gsce'], name = 'tce')

	return model
