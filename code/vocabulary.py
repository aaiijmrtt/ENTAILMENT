import argparse
import datetime
import sys
import os

def filtertext(char):
	return char.isdigit() or char.isalpha() or char == ' '

def maptext(char):
	if char == '0': return 'zero'
	elif char == '1': return 'one'
	elif char == '2': return 'two'
	elif char == '3': return 'three'
	elif char == '4': return 'four'
	elif char == '5': return 'five'
	elif char == '6': return 'six'
	elif char == '7': return 'seven'
	elif char == '8': return 'eight'
	elif char == '9': return 'nine'
	return char.lower()

def count(filename, vocab = dict()):
	for line in open(filename):
		label, premise, hypothesis = line.split('\t')
		for word in map(maptext, filter(filtertext, premise.split())):
			if word in vocab: vocab[word] += 1
			else: vocab[word] = 1
		for word in map(maptext, filter(filtertext, hypothesis.split())):
			if word in vocab: vocab[word] += 1
			else: vocab[word] = 1
	return vocab

def writetofile(filenamein, filenameout, vocab):
	with open(filenameout, 'w') as fileout:
		for line in open(filenamein):
			label, premise, hypothesis = line.split('\t')
			if label == 'neutral': fileout.write(str(0) + '\t')
			elif label == 'contradiction': fileout.write(str(1) + '\t')
			elif label == 'entailment': fileout.write(str(2) + '\t')
			else: continue
			for word in map(maptext, filter(filtertext, premise.split())):
				fileout.write(str(vocab.index(word) + 1) + ' ')
			fileout.write('\t')
			for word in map(maptext, filter(filtertext, hypothesis.split())):
				fileout.write(str(vocab.index(word) + 1) + ' ')
			fileout.write('\n')

def main():
	processed_data_path = os.path.abspath('../code/data')
	parser = argparse.ArgumentParser(description = 'Build vocabulary of preprocessed files')
	parser.add_argument('-d', '--dir', action = 'store', default = processed_data_path, 
			    type = str, help = 'Root directory of preprocessed splits')
	args = parser.parse_args()

	vocab = count(os.path.join(args.dir, 'snli_processed_dev.txt'))
	print datetime.datetime.now(), 'Read dev'
	vocab = count(os.path.join(args.dir, 'snli_processed_test.txt'), vocab)
	print datetime.datetime.now(), 'Read test'
	vocab = count(os.path.join(args.dir, 'snli_processed_train.txt'), vocab)
	print datetime.datetime.now(), 'Read train'
	vocab = map(lambda x: x[0], sorted(vocab.items(), key = lambda x: x[1], reverse = True))
	writetofile(os.path.join(args.dir, 'snli_processed_dev.txt'), 
		    os.path.join(args.dir, 'snli_mapped_dev.txt'), vocab)
	print datetime.datetime.now(), 'Write dev'
	writetofile(os.path.join(args.dir, 'snli_processed_test.txt'), 
		    os.path.join(args.dir, 'snli_mapped_test.txt'), vocab)
	print datetime.datetime.now(), 'Write test'
	writetofile(os.path.join(args.dir, 'snli_processed_train.txt'), 
		    os.path.join(args.dir, 'snli_mapped_train.txt'), vocab)
	print datetime.datetime.now(), 'Write train'

if __name__ == '__main__':
	main()
