import numpy as np
import gensim
from gensim.models import KeyedVectors
import sys
import argparse
import json
from scipy import stats
from gensim import matutils
from scipy.spatial import distance
import re
from nltk.corpus import wordnet as wn


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--embedding', type=str, required=False)
	parser.add_argument('--output', type=str, required=True)
	parser.add_argument('--seed', type=int, default=1111)

	args = parser.parse_args()

	return args


def load_lmms(npz_vecs_path):
	lmms = {}
	loader = np.load(npz_vecs_path)
	labels = loader['labels'].tolist()
	vectors = loader['vectors']
	for label, vector in list(zip(labels, vectors)):
		lmms[label] = vector
	return lmms


def load_ares_txt(path):
    sense_vecs = {}
    with open(path, 'r') as sfile:
        for idx, line in enumerate(sfile):
            if idx == 0:
                continue
            splitLine = line.split(' ')
            label = splitLine[0]
            vec = np.array(splitLine[1:], dtype=float)
            dim = vec.shape[0]
            sense_vecs[label] = vec
    return sense_vecs


def get_sk_lemma(sensekey):
	return sensekey.split('%')[0]


def word_assoc(w, a_attr_list, b_attr_list, sense_emb):	
	return n_similarity(sense_emb, a_attr_list) - n_similarity(sense_emb,b_attr_list)


def diff_assoc(X, Y, A, B, sense_emb):
	sense_id_list = sense_emb.keys()

	### Convert relevant sense lists for words in each of the x, y, a, b lists
	x_senses = get_relevent_sense_list(sense_id_list, X)
	y_senses = get_relevent_sense_list(sense_id_list, Y)
	a_senses = get_relevent_sense_list(sense_id_list, A)
	b_senses = get_relevent_sense_list(sense_id_list, B)

	xa_target_attr_list = get_target_attr(x_senses, a_senses, sense_emb)
	xb_target_attr_list = get_target_attr(x_senses, b_senses, sense_emb)
	ya_target_attr_list = get_target_attr(y_senses, a_senses, sense_emb)
	yb_target_attr_list = get_target_attr(y_senses, b_senses, sense_emb)

	word_assoc_X = np.array(list(map(lambda x : word_assoc(x, xa_target_attr_list, xb_target_attr_list, sense_emb), X)))
	word_assoc_Y = np.array(list(map(lambda y : word_assoc(y, ya_target_attr_list, yb_target_attr_list, sense_emb), Y)))
	mean_diff = np.mean(word_assoc_X) - np.mean(word_assoc_Y)
	std = np.std(np.concatenate((word_assoc_X, word_assoc_Y), axis=0))
	# print('mean_diff: ', mean_diff, 'std: ', std)
	return mean_diff / std


def random_choice(word_pairs, subset_size):
	return np.random.choice(word_pairs,
							subset_size,
							replace=False)


def get_bias_scores_mean_err(word_pairs, sense_emb):
	print('word pairs: ', word_pairs)
	subset_size_target = min(len(word_pairs['X']), len(word_pairs['Y'])) // 2
	subset_size_attr = min(len(word_pairs['A']), len(word_pairs['B'])) // 2
	bias_scores = [diff_assoc(
		random_choice(word_pairs['X'], subset_size_target),
		random_choice(word_pairs['Y'], subset_size_target),
		random_choice(word_pairs['A'], subset_size_attr),
		random_choice(word_pairs['B'], subset_size_attr), sense_emb) for _ in range(5000)]
	# print('subset_size_target: ', subset_size_target, 'subset_size_attr: ', subset_size_attr)
	# print('bias_scores: ', bias_scores)
	bias_scores = [x for x in bias_scores if np.isnan(x) == False]
	# print('after remove nan: ', bias_scores)
	return np.mean(bias_scores), stats.sem(bias_scores)



def n_similarity(emb, target_attr_list):
		"""
		Compute cosine similarity between two sets of words.

		Example::

		  >>> trained_model.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])
		  0.61540466561049689

		  >>> trained_model.n_similarity(['restaurant', 'japanese'], ['japanese', 'restaurant'])
		  1.0000000000000004

		  >>> trained_model.n_similarity(['sushi'], ['restaurant']) == trained_model.similarity('sushi', 'restaurant')
		  True

		"""
		# print('target_attr_list', target_attr_list)
		v1 = [emb[i[0]] for i in target_attr_list]
		v2 = [emb[i[1]] for i in target_attr_list]

		return np.dot(matutils.unitvec(np.array(v1).mean(axis=0)), matutils.unitvec(np.array(v2).mean(axis=0)))


def get_relevent_sense_list(sense_id_list, word_list):
	word_senses = []
	for word in word_list:
		word_sks = []
		for sense_id in sense_id_list:
			if word == get_sk_lemma(sense_id):
				word_sks.append(sense_id)
		word_senses.append(word_sks)
	return word_senses


def get_target_attr(target_senses, attr_senses, sense_emb):
	### select sense embeddings for pair of words with the maximum similarity score
	target_attr_list = []
	for target_sks in target_senses:
		for attr_sks in attr_senses:    
			max_sim_target_attr = 0   
			for target_sk in target_sks: 
				for attr_sk in attr_sks:
					sim_target_attr = 1 - distance.cosine(sense_emb[target_sk], sense_emb[attr_sk])     
					if sim_target_attr > max_sim_target_attr:
						max_sim_target_attr = sim_target_attr
						max_target_sk = target_sk
						max_attr_sk = attr_sk
			target_attr_list.append((max_target_sk, max_attr_sk))

	return target_attr_list


def run_test(config, sense_emb):
	word_pairs = {}
	sense_lemma = {}
	min_len = sys.maxsize

	for sense in sense_emb.keys():
		lemma = get_sk_lemma(sense)
		sense_lemma[sense] = lemma

	for word_list_name, word_list in config.items():
		if word_list_name in ['X', 'Y', 'A', 'B']:
			word_list_filtered = list(filter(lambda x: x in sense_lemma.values(), word_list))
			word_pairs[word_list_name] = word_list_filtered
			if len(word_list_filtered) < 2:
				print('ERROR: Words from list {} not found in embedding\n {}'.\
				format(word_list_name, word_list))
				print('All word groups must contain at least two words')
				return None, None
	return get_bias_scores_mean_err(word_pairs, sense_emb)


def eval_weat(sense_emb, output):
	config = json.load(open('data/weat-test.json'))
	with open(output, 'w') as fw:
			for name_of_test, test_config in config['tests'].items():
				score, err  = run_test(test_config, sense_emb)
				print('name_of_test: ', name_of_test, 'score: ', score)
				if score is not None:
					score = str(round(score, 4))
					err = str(round(err, 4))
					fw.write(f'{name_of_test}\n')
					fw.write(f'Score: {score}\n')
					fw.write(f'P-value: {err}\n')
			

def main(args):
	sense_emb = load_lmms('path to lmms embeddings')
	# sense_emb = load_ares_txt("path to ares embeddings")
	eval_weat(sense_emb, args.output)


if __name__ == '__main__':
	args= parse_args()
	np.random.seed(args.seed)
	main(args)

