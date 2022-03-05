import numpy as np
import argparse
from gensim.models import KeyedVectors
from scipy.stats import pearsonr
from scipy.spatial import distance


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--embedding', type=str, required=False)
	parser.add_argument('--output', type=str, required=True)

	args = parser.parse_args()

	return args


def cos_sim(v1, v2):
	return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


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
			sim_target_attr = 1 - distance.cosine(sense_emb[target_sks], sense_emb[attr_sks])     
			if sim_target_attr > max_sim_target_attr:
				max_sim_target_attr = sim_target_attr
				max_target_sk = target_sks
				max_attr_sk = attr_sks
	target_attr_list.append((max_target_sk, max_attr_sk))

	return target_attr_list


def eval_wat(emb, output, path, word1_list, word2_list):
	word2sense = {}
	for sense in emb.keys():
		lemma = get_sk_lemma(sense)
		word2sense[lemma] = sense

	gold_d = {}
	for l in open(path):
		word, score = l.strip().split('\t')
		if word in word2sense.keys():
			gold_d[word] = float(score)
	emb_d = dict()
	for key in gold_d.keys():
		if key in word1_list or key in word2_list:
			continue
		relevant_sks = []
		for sense in emb.keys():
			if key == get_sk_lemma(sense):
				relevant_sks.append(sense)

		if len(relevant_sks)==0:
			continue

		scores_list = []
		for w1, w2 in zip(word1_list, word2_list):
			relevant_sks_1 = []
			relevant_sks_2 = []
			for sense in emb.keys():
				if w1 == get_sk_lemma(sense):
					relevant_sks_1.append(sense)
				if w2 == get_sk_lemma(sense):
					relevant_sks_2.append(sense)

			if len(relevant_sks_1)==0 or len(relevant_sks_2)==0:
				continue

			target_attr_list_w1 = get_target_attr(relevant_sks, relevant_sks_1, emb)
			target_attr_list_w2 = get_target_attr(relevant_sks, relevant_sks_2, emb)
			print('-----target_attr_list_w1',target_attr_list_w1, '-------target_attr_list_w2', target_attr_list_w2)

			for e1, s_1 in target_attr_list_w1:
				print('e1', e1, 's_1', s_1)
				score1 = cos_sim(emb[e1], emb[s_1])
			for e2, s_2 in target_attr_list_w2:
				print('e2', e2, 's_2', s_2)
				score2 = cos_sim(emb[e2], emb[s_2])
			score = score1 - score2
			scores_list.append(score)

		score = sum(scores_list) / len(scores_list)
		emb_d[key] = score

	g_l = []
	e_l = []
	for key in emb_d.keys():
		g_l += [gold_d[key]]
		e_l += [emb_d[key]]
	r, p = pearsonr(np.array(g_l), np.array(e_l))

	with open(output, 'w') as fw:
		fw.write(f'Pearson’s correlation coefficient: {r}\n')
		fw.write(f'P-value: {p}\n')


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
            # print('self.dim', self.dim)
            sense_vecs[label] = vec
    return sense_vecs


def get_sk_lemma(sensekey):
	return sensekey.split('%')[0]


def main(args):
	sense_emb = load_lmms('path to lmms embeddings')
	# sense_emb = load_ares_txt("path to ares embeddings")

	word1_list = ['he', 'father', 'son', 'husband', 'grandfather',
				  'brother', 'man', 'boy', 'uncle', 'gentleman']
	word2_list = ['she', 'mother', 'daughter', 'wife', 'grandmother',
				  'sister', 'woman', 'girl', 'aunt', 'lady']

	eval_wat(sense_emb, args.output, 'data/wat_bi.txt', word1_list, word2_list)

if __name__ == "__main__":
	args = parse_args()
	main(args)
