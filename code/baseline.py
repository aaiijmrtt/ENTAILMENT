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
					model['px_%i_%i' %(i, ii)] = model['px_%i' %ii] if i == 0 else model['ph_%i_%i' %(i - 1, ii)]

			with tf.name_scope('inputgate_%i' %i):
				model['pWi_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pWi_%i' %i)
				model['pBi_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBi_%i' %i)
				for ii in xrange(dim_t):
					model['pi_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['px_%i_%i' %(i, ii)], model['pWi_%i' %i]), model['pBi_%i' %i]), name = 'pi_%i_%i' %(i, ii))

			with tf.name_scope('forgetgate_%i' %i):
				model['pWf_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pWf_%i' %i)
				model['pBf_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBf_%i' %i)
				for ii in xrange(dim_t):
					model['pf_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['px_%i_%i' %(i, ii)], model['pWf_%i' %i]), model['pBf_%i' %i]), name = 'pf_%i_%i' %(i, ii))

			with tf.name_scope('outputgate_%i' %i):
				model['pWo_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pWo_%i' %i)
				model['pBo_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBo_%i' %i)
				for ii in xrange(dim_t):
					model['po_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['px_%i_%i' %(i, ii)], model['pWo_%i' %i]), model['pBo_%i' %i]), name = 'po_%i_%i' %(i, ii))

			with tf.name_scope('cellstate_%i' %i):
				model['pWc_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pWc_' + str(i))
				model['pBc_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBc_' + str(i))
				for ii in xrange(dim_t):
					model['pcc_%i_%i' %(i, ii)] = tf.Variable(tf.truncated_normal([dim_b, dim_i], stddev = 1.0 / dim_i), name = 'pcc_%i_%i' %(i, ii)) if ii == 0 else model['pc_%i_%i' %(i, ii - 1)] # consider starting with all zeros
					model['pc_%i_%i' %(i, ii)] = tf.add(tf.multiply(model['pf_%i_%i' %(i, ii)], model['pcc_%i_%i' %(i, ii)]), tf.multiply(model['pi_%i_%i' %(i, ii)], tf.nn.tanh(tf.add(tf.matmul(model['px_%i_%i' %(i, ii)], model['pWc_%i' %i]), model['pBc_%i' %i]))), name = 'pc_%i_%i' %(i, ii))

			with tf.name_scope('hidden_%i' %i):
				model['pWz_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'pWz_%i' %i)
				model['pBz_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'pBz_%i' %i)
				for ii in xrange(dim_t):
					model['pz_%i_%i' %(i, ii)] = tf.add(tf.matmul(model['pc_%i_%i' %(i, ii)], model['pWz_%i' %i]), model['pBz_%i' %i], name = 'pz_%i_%i' %(i, ii))

			with tf.name_scope('output_%i' %i):
				for ii in xrange(dim_t):
					model['ph_%i_%i' %(i, ii)] = tf.multiply(model['po_%i_%i' %(i, ii)], tf.nn.tanh(model['pz_%i_%i' %(i, ii)]), name = 'ph_%i_%i' %(i, ii))

		with tf.name_scope('output'):
			for ii in xrange(dim_t):
				model['ph_%i' %ii] = model['ph_%i_%i' %(dim_d - 1, ii)]

		with tf.name_scope('meansquared'):
			for ii in xrange(dim_t):
				model['pms_%i' %ii] = tf.where(tf.equal(model['pxi_%i' %ii], tf.zeros([dim_b], tf.int32)), tf.zeros([dim_b], tf.float32), tf.reduce_sum(tf.square(tf.subtract(model['py_%i' %ii], model['ph_%i' %ii])), [1]), name = 'pms_%i' %ii)
			model['pms'] = tf.reduce_sum(tf.add_n([model['pms_%i' %ii] for ii in xrange(dim_t)]), name = 'pms')
			model['spms'] = tf.summary.scalar(model['pms'].name, model['pms'])

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
					model['hx_%i_%i' %(i, ii)] = model['hx_%i' %ii] if i == 0 else model['hh_%i_%i' %(i - 1, ii)]

			with tf.name_scope('inputgate_%i' %i):
				model['hWi_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'hWi_%i' %i)
				model['hBi_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'hBi_%i' %i)
				for ii in xrange(dim_t):
					model['hi_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['hx_%i_%i' %(i, ii)], model['hWi_%i' %i]), model['hBi_%i' %i]), name = 'hi_%i_%i' %(i, ii))

			with tf.name_scope('forgetgate_%i' %i):
				model['hWf_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'hWf_%i' %i)
				model['hBf_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'hBf_%i' %i)
				for ii in xrange(dim_t):
					model['hf_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['hx_%i_%i' %(i, ii)], model['hWf_%i' %i]), model['hBf_%i' %i]), name = 'hf_%i_%i' %(i, ii))

			with tf.name_scope('outputgate_%i' %i):
				model['hWo_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'hWo_%i' %i)
				model['hBo_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'hBo_%i' %i)
				for ii in xrange(dim_t):
					model['ho_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['hx_%i_%i' %(i, ii)], model['hWo_%i' %i]), model['hBo_%i' %i]), name = 'ho_%i_%i' %(i, ii))

			with tf.name_scope('cellstate_%i' %i):
				model['hWc_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'hWc_' + str(i))
				model['hBc_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'hBc_' + str(i))
				for ii in xrange(dim_t):
					model['hcc_%i_%i' %(i, ii)] = tf.Variable(tf.truncated_normal([dim_b, dim_i], stddev = 1.0 / dim_i), name = 'hcc_%i_%i' %(i, ii)) if ii == 0 else model['hc_%i_%i' %(i, ii - 1)] # consider starting with all zeros
					model['hc_%i_%i' %(i, ii)] = tf.add(tf.multiply(model['hf_%i_%i' %(i, ii)], model['hcc_%i_%i' %(i, ii)]), tf.multiply(model['hi_%i_%i' %(i, ii)], tf.nn.tanh(tf.add(tf.matmul(model['hx_%i_%i' %(i, ii)], model['hWc_%i' %i]), model['hBc_%i' %i]))), name = 'hc_%i_%i' %(i, ii))

			with tf.name_scope('hidden_%i' %i):
				model['hWz_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'hWz_%i' %i)
				model['hBz_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'hBz_%i' %i)
				for ii in xrange(dim_t):
					model['hz_%i_%i' %(i, ii)] = tf.add(tf.matmul(model['hc_%i_%i' %(i, ii)], model['hWz_%i' %i]), model['hBz_%i' %i], name = 'hz_%i_%i' %(i, ii))

			with tf.name_scope('output_%i' %i):
				for ii in xrange(dim_t):
					model['hh_%i_%i' %(i, ii)] = tf.multiply(model['ho_%i_%i' %(i, ii)], tf.nn.tanh(model['hz_%i_%i' %(i, ii)]), name = 'hh_%i_%i' %(i, ii))

		with tf.name_scope('output'):
			for ii in xrange(dim_t):
				model['hh_%i' %ii] = model['hh_%i_%i' %(dim_d - 1, ii)]

		with tf.name_scope('meansquared'):
			for ii in xrange(dim_t):
				model['hms_%i' %ii] = tf.where(tf.equal(model['hxi_%i' %ii], tf.zeros([dim_b], tf.int32)), tf.zeros([dim_b], tf.float32), tf.reduce_sum(tf.square(tf.subtract(model['hy_%i' %ii], model['hh_%i' %ii])), [1]), name = 'hms_%i' %ii)
			model['hms'] = tf.reduce_sum(tf.add_n([model['hms_%i' %ii] for ii in xrange(dim_t)]), name = 'hms')
			model['shms'] = tf.summary.scalar(model['hms'].name, model['hms'])

	with tf.name_scope('classification'):
		with tf.name_scope('label'):
			model['clabel'] = tf.placeholder(tf.float32, [dim_b, dim_c], name = 'clabel')

		for i in xrange(dim_n):
			with tf.name_scope('layer_%i' %i):
				model['cW_%i' %i] = tf.Variable(tf.truncated_normal([2 * dim_i, 2 * dim_i], stddev = 0.5 / dim_i), name = 'cW_%i' %i) if i != dim_n - 1 else tf.Variable(tf.truncated_normal([2 * dim_i, dim_c], stddev = 1.0 / dim_c), name = 'cW_%i' %i)
				model['cB_%i' %i] = tf.Variable(tf.truncated_normal([1, 2 * dim_i], stddev = 0.5 / dim_i), name = 'cB_%i' %i) if i != dim_n - 1 else tf.Variable(tf.truncated_normal([1, dim_c], stddev = 1.0 / dim_c), name = 'cB_%i' %i)
				model['cx_%i' %i] = tf.concat(axis=1, values=[model['ph_%i' %(dim_t - 1)], model['hh_%i' %(dim_t - 1)]], name = 'cx_%i' %i) if i == 0 else model['cy_%i' %(i - 1)]
				model['cy_%i' %i] = tf.add(tf.matmul(model['cx_%i' %i], model['cW_%i' %i]), model['cB_%i' %i], name = 'cy_%i' %i)

		with tf.name_scope('output'):
			model['output'] = tf.nn.softmax(model['cy_%i' %(dim_n - 1)], name = 'output')

		with tf.name_scope('crossentropy'):
			model['cce'] = tf.reduce_sum(-tf.multiply(model['clabel'], tf.log(model['output'])), name = 'cce')
			model['scce'] = tf.summary.scalar(model['cce'].name, model['cce'])

	model['gsms'] = tf.Variable(0, trainable = False, name = 'gsms')
	model['lrms'] = tf.train.exponential_decay(lrate_ms, model['gsms'], dstep_ms, drate_ms, staircase = False, name = 'lrms')
	model['tms'] = optim_ms(model['lrms']).minimize(model['pms'] + model['hms'], global_step = model['gsms'], name = 'tms')

	model['gsce'] = tf.Variable(0, trainable = False, name = 'gsce')
	model['lrce'] = tf.train.exponential_decay(lrate_ce, model['gsce'], dstep_ce, drate_ce, staircase = False, name = 'lrce')
	model['tce'] = optim_ce(model['lrce']).minimize(model['cce'], global_step = model['gsce'], name = 'tce')

	return model
