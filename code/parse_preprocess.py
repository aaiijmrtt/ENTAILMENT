import argparse
import json
import os

def iter_parse(file_obj):
	decoder = json.JSONDecoder()
	buff = ""
	for line in file_obj:
		buff += line.strip()
		try:
			result = decoder.raw_decode(buff)
			buff = ""
			yield result[0]
		except ValueError:
			pass

def parse_json(filename):
	SNLI = []
	with open(filename) as f:
		for obj in iter_parse(f):
			SNLI.append(obj)
	obj = None
	del obj
	return SNLI

def write_file(filename, SNLI):
	with open(os.path.join(processed_data_path, filename), 'w') as f:
		for idx in xrange(len(SNLI)):
			label = SNLI[idx]['gold_label']
			premise = SNLI[idx]['sentence1']
			hypothesis = SNLI[idx]['sentence2']
			line = label + '\t' + premise + '\t' + hypothesis + '\n'
			f.write(line)

def main():
	global default_SNLI_path, processed_data_path, label
	default_SNLI_path = os.path.abspath('../snli_1.0/snli_1.0')
	processed_data_path = os.path.abspath('../code/data')

	parser = argparse.ArgumentParser(description = "Preprocess train, dev, test JSON splits")
	parser.add_argument('-j', '--json_dir', action = "store", default = default_SNLI_path, type = str,
						help = "Root directory path of JSON splits")
	parser.add_argument('-s', '--split', nargs = '+', action = "store", required = True, type = str,
						help = "Preprocess splits Options: train, dev, test")
	args = parser.parse_args()

	dir_path = args.json_dir
	train_file = dir_path.rstrip('/') + '/' + 'snli_1.0_train.jsonl'
	dev_file = dir_path.rstrip('/') + '/' + 'snli_1.0_dev.jsonl'
	test_file = dir_path.rstrip('/') + '/' + 'snli_1.0_test.jsonl'

	if not os.path.exists(processed_data_path):
		os.makedirs(processed_data_path)

	if ('train' not in args.split) and ('dev' not in args.split) and ('test' not in args.split):
		raise argparse.ArgumentTypeError('Argument has to be from:  train dev test')

	else:
		if 'train' in args.split:
			SNLI = parse_json(train_file)
			filename = 	'snli_processed_train.txt'
			write_file(filename, SNLI)

		if 'dev' in args.split:
			SNLI = parse_json(dev_file)
			filename = 	'snli_processed_dev.txt'
			write_file(filename, SNLI)
    
		if 'test' in args.split:
			SNLI = parse_json(train_file)
			filename = 	'snli_processed_test.txt'
			write_file(filename, SNLI)

if __name__ == '__main__':
	main()