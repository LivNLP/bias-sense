"""
This script contains methods for conducting evaluation of biases
in sense-sensitive word embeddings using the datasets released.
"""

import numpy as np
from scipy import stats
import random
import pandas as pd

import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet as wn

def find_sense_id(word):
    """
    Print the sense ids of words.
    """
    synsets = wn.synsets(word)
    for x in synsets:
        print(x.lemmas()[0].key(), x.pos(), "===", x.definition())


def cosine(x,y):
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    return (np.dot(x,y) / norm) if norm > 0 else 0


def sample_and_average(L, x):
    """
    L is a list of vectors, all representing a particular type of bias.
    x is the embedding of a word that we would like to evaluate for its bias.
    We will subsample vectors from L, compute the mean and measure cosine similarity with x.
    We will then compute a sigificance score based on these multiple similarities.
    """
    sample_size = len(L) // 2
    scores = []
    for _ in range(5000):
        idx = np.random.choice(len(L), size=sample_size)
        diff = [cosine(x, L[i]) for i in idx]
        mean_diff = np.mean(diff)
        sd_diff = np.std(diff)
        scores.append(mean_diff / sd_diff)
    return np.mean(scores), stats.sem(scores)


def two_sided_sampling(x, positives, negatives):
    """
    Sample equal size adjective sets from positive and negative adjectives.
    Measure the average cosine similarity between the target sense and each sample.
    Compute the difference of the similarity between positive and negative adjective sets.
    Compute the mean and standard error on these differences.
    If the mean difference is zero, then there is no ethnic bias.
    """
    bias_scores = []
    for _ in range(5000):
        t = min(len(positives), len(negatives))
        sample_size = random.randint(t // 2, t)
        pos_idx = np.random.choice(len(positives), size=sample_size)
        pos_scores = [cosine(x, positives[i]) for i in pos_idx]
        pos_score = np.mean(pos_scores) / np.std(pos_scores)
        neg_idx = np.random.choice(len(negatives), size=sample_size)
        neg_scores = [cosine(x, negatives[i]) for i in neg_idx]
        neg_score = np.mean(neg_scores) / np.std(neg_scores)
        bias_scores.append(neg_score - pos_score)
    return np.mean(bias_scores), stats.sem(bias_scores)

def load_positives_negatives(WE):
    """
    Load positive and negative adjectives from files. 
    We will consider the first adjectivial sense for each word.
    """
    positives = []
    with open("./data/positive-adjectives") as pos_file:
        for line in pos_file:
            word = line.strip()
            synsets = wn.synsets(word)
            adj_synsets = []
            for synset in synsets:
                sid = synset.lemmas()[0].key()
                pos = int(sid.split("%")[1].split(':')[0])
                if pos == 3:
                    adj_synsets.append(sid)
            if len(adj_synsets) == 0:
                continue
            adj_sid = adj_synsets[0]
            if adj_sid not in WE.embed:
                continue
            positives.append(WE.get_vector(adj_sid))
    print("Total number of positive adjectives =", len(positives))

    negatives = []
    with open("./data/negative-adjectives") as neg_file:
        for line in neg_file:
            word = line.strip()
            synsets = wn.synsets(word)
            adj_synsets = []
            for synset in synsets:
                sid = synset.lemmas()[0].key()
                pos = int(sid.split("%")[1].split(':')[0])
                if pos == 3:
                    adj_synsets.append(sid)
            if len(adj_synsets) == 0:
                continue
            adj_sid = adj_synsets[0]
            if adj_sid not in WE.embed:
                continue
            negatives.append(WE.get_vector(adj_sid))
    print("Total number of negative adjectives =", len(negatives))
    return positives, negatives


def eval_ethnic_bias(WE):
    """
    Evaluate the black as ethnic group vs. colour bias.
    """
    balck_ethnic_sid = "black%1:18:00::"
    black_colour_sid = "black%1:07:00::"

    positives, negatives = load_positives_negatives(WE)    
      
    if black_colour_sid not in WE.embed:
        print("Colour sense of black missing")
        raise ValueError

    if balck_ethnic_sid not in WE.embed:
        print("Ethnic sense of black missing")
        raise ValueError
    
    colour_bias, colour_err = two_sided_sampling(WE.get_vector(black_colour_sid), positives, negatives)
    ethnic_bias, ethnic_err = two_sided_sampling(WE.get_vector(balck_ethnic_sid), positives, negatives)
    res = {"black" : {"colour_bias":colour_bias, "colour_err":colour_err, 
                      "ethnic_bias":ethnic_bias, "ethnic_err":ethnic_err}}
    df = pd.DataFrame(data=res)
    print(df.T)
    return df


def eval_racial_bias(WE):
    """
    Evaluatte nationalities vs. languages.
    """
    nationalities = ["Japanese", "Chinese", "English", "Arabic", "German",
                     "French", "Spanish", "Portuguese", "Norwegian", "Swedish", "Polish", "Romanian",
                     "Russian", "Egyptian", "Finnish", "Vietnamese"]
    
    people_sid_suffix = "%1:18:00::"
    lang_sid_suffix = "%1:10:00::"

    positives, negative = load_positives_negatives(WE)  
    res = {}
    for nation in nationalities:
        people_sid = "%s%s" % (nation.lower(), people_sid_suffix)
        lang_sid = "%s%s" % (nation.lower(), lang_sid_suffix)
        both_senses_found = True
        if people_sid not in WE.embed:
            print("People sense of {0} not found!".format(nation))
            both_senses_found = False
        if lang_sid not in WE.embed:
            print("Language sense of {0} not found!".format(nation))
            both_senses_found = False
        if not both_senses_found:
            print("Skipping {0}".format(nation))
            continue
        res[nation] = {}
        people_bias_score, people_bias_err = two_sided_sampling(WE.get_vector(people_sid), positives, negatives)
        res[nation] = {"people_bias":people_bias_score, "people_err":people_bias_err}
        lang_bias_score, lang_bias_err = two_sided_sampling(WE.get_vector(lang_sid), positives, negatives)
        res[nation] = {"lang_bias":lang_bias_score, "lang_err":lang_bias_err}
    df = pd.DataFrame(data=res)
    print(df.T)
    return df     


def eval_gender_bias(WE):
    """
    Evaluate gender bias, where we first define the gender direction by the vector offset of
    word-pairs describing male vs. female attributes. We will then evaluate noun and verb senses
    of a list of target words and return their individual and aggregated scores with statistical
    significance scores (evaluated according to a boostrapping test).
    """
    male_words = []
    with open("./data/male_word_file.txt") as male_file:
        for line in male_file:
            male_words.append(line.strip())
    female_words = []
    with open("./data/female_word_file.txt") as female_file:
        for line in female_file:
            female_words.append(line.strip())
    gender_pairs = list(zip(male_words, female_words))
    gender_vects = []

    for (male, female) in gender_pairs:
        #print(male,female)
        male_synset = wn.synsets(male)
        female_synset = wn.synsets(female)
        if len(male_synset) == 0 or len(female_synset) == 0:
            continue
        male_sid = male_synset[0].lemmas()[0].key()
        female_sid = female_synset[0].lemmas()[0].key()
        #print(male_sid, female_sid)
        gender_vects.append(WE.get_vector(male_sid) - WE.get_vector(female_sid))
    #print("Total number of gender word-pairs =", len(gender_pairs))

    occupations = [("engineer", "engineer%1:18:00::", "engineer%2:31:01::"),
                   ("carpenter", "carpenter%1:18:00::", "carpenter%2:41:00::"), 
                   ("guide", "guide%1:18:00::", "guide%2:38:00::"),
                   ("mentor", "mentor%1:18:00::", "mentor%2:32:00::"),
                   ("judge", "judge%1:18:00::", "judge%2:31:02::"),
                   ("nurse", "nurse%1:18:00::", "nurse%2:29:00::")]
    
    res = {}
    for (word, noun_sid, verb_sid) in occupations:
        res[word] = {}
        if noun_sid not in WE.embed:
            print("Noun Sense Embedding Not Found for =", word)
            bias_score, bias_error = 0, 0
        else:
            noun_emb = WE.get_vector(noun_sid)
            bias_score, bias_error = sample_and_average(gender_vects, noun_emb)
        res[word]["noun_bias"] = bias_score
        res[word]["noun_err"] = bias_error

        if verb_sid not in WE.embed:
            print("Verb Sense Embedding Not Found for =", word)
            bias_score, bias_error = 0, 0
        else:
            verb_emb = WE.get_vector(verb_sid)
            bias_score, bias_error = sample_and_average(gender_vects, verb_emb)
        res[word]["verb_bias"] = bias_score
        res[word]["verb_err"] = bias_error        
        pass
    df = pd.DataFrame(data=res)
    avg = df.copy()
    avg['mean'] = df.T.mean(numeric_only=1)
    print(avg)
    return avg

class WordEmbedding(object):

    def __init__(self, fname):
        """
        Load the word embeddings from fname.
        """
        self.embed = self.load_lmms(fname)
        print("Total number of vectors =", len(self.embed))
        pass

    def load_lmms(self, npz_vecs_path):
        lmms = {}
        loader = np.load(npz_vecs_path)
        labels = loader['labels'].tolist()
        vectors = loader['vectors']
        self.dim = vectors[0].shape[0]
        for label, vector in list(zip(labels, vectors)):
            lmms[label] = vector
        return lmms

    def get_vector(self, label):
        """
        If the label is not a sense-id (i.e. in the case of sense-insensitive static word embeddings)
        return the word embedding instead of sense embedding. You will need to modify this function 
        according to the word embedding you want to evaluate. If the word is not in the sense embedding
        return a zero vector of the same dimensionality.
        """
        return self.embed.get(label, np.zeros(self.dim))


def main():
    WE = WordEmbedding("./data/lmms_1024.bert-large-cased.npz")    
    #eval_ethnic_bias(WE)
    eval_racial_bias(WE)
    eval_gender_bias(WE)
    pass

def debug():
    L = np.array([[1,2],[3,4], [5,6]])
    for i in range(10):
        idx = np.random.choice(len(L), size=2, replace=False)
        print(L[idx,:])
        #print(L[np.random.choice(range(3),size=2)])
        #print(random.choice(L, 2))
   

if __name__ == "__main__":
    #find_sense_id("dirty")
    main()
    #debug()