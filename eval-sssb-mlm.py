import json
import argparse
import torch
import difflib
import sys
import tensorflow as tf
import numpy as np

from tqdm import tqdm
from scipy import stats
from collections import defaultdict
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sensebert import SenseBert


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        choices=['cp', 'ss',
                            'sssb_gender', 'sssb_race', 'sssb_nationality'],
                        help='Path to evaluation dataset.')
    parser.add_argument('--model', type=str, required=True,
                        choices=['sense-base', 'sense-large',
                                 'bert-base', 'bert-large'])
    parser.add_argument('--method', type=str, required=True,
                        choices=['aul', 'cps', 'sss'])
    args = parser.parse_args()

    return args


def load_tokenizer_and_model(args):
    '''
    Load tokenizer and model to evaluate.
    '''
    if args.model == 'sense-base':
        pretrained_weights = 'sensebert-base-uncased'
    elif args.model == 'sense-large':
        pretrained_weights = 'sensebert-large-uncased'
    elif args.model == 'bert-base':
        pretrained_weights = 'bert-base-uncased'
    elif args.model == 'bert-large':
        pretrained_weights = 'bert-large-uncased'

    if args.model == 'sense-base' or args.model == 'sense-large':
        config = tf.ConfigProto()
        config.gpu_options.allow_growth =True
        session = tf.Session(config=config)
        # sensebert_model = sensebert.SenseBert(pretrained_weights, session=session)
        sensebert_model = SenseBert(pretrained_weights, session=session)
        
        
        return sensebert_model.tokenize, sensebert_model.run

    elif args.model == 'bert-base' or args.model == 'bert-large':
        model = AutoModelForMaskedLM.from_pretrained(pretrained_weights,
                                                     output_hidden_states=True,
                                                     output_attentions=True)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)
        model = model.eval()

        return tokenizer, model


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)


def calculate_aul_for_sense_bert(model, token_ids, mask_ids):
    '''
    Given token ids of a sequence, return the averaged log probability of
    unmasked sequence (AUL).
    '''
    embeddings, token_logits, sense_logits = model(token_ids, mask_ids)
    token_ids = np.array(token_ids)
    token_ids = token_ids.reshape(-1)
    token_logits = token_logits.squeeze()
    # token_logits = softmax(token_logits)
    token_logits = np.log(softmax(token_logits))
    log_prob = np.mean(np.take(token_logits, token_ids, 1)[1:-1])
    log_prob = log_prob.item()

    return log_prob


def calculate_aul_for_bert(model, token_ids, log_softmax):
    '''
    Given token ids of a sequence, return the averaged log probability of
    unmasked sequence (AUL).
    '''
    output = model(token_ids)
    logits = output.logits.squeeze(0)
    log_probs = log_softmax(logits)
    token_ids = token_ids.view(-1, 1).detach()
    token_log_probs = log_probs.gather(1, token_ids)[1:-1]
    log_prob = torch.mean(token_log_probs)
    log_prob = log_prob.item()

    return log_prob



def main(args):
    '''
    Evaluate the bias in masked language models.
    '''
    tokenizer, model = load_tokenizer_and_model(args)
    log_softmax = torch.nn.LogSoftmax(dim=1)
    total_score = 0
    stereo_score = 0

    mask_id = 103
    counts = defaultdict(int)
    scores = defaultdict(int)
    data = []
    bias_scores = []
    mlm_scores = []

    with open('data/' + args.data + '.json') as f:
        inputs = json.load(f)
        total_num = len(inputs)
        for input in tqdm(inputs):
            if args.data == 'cp' or args.data == 'ss':
                bias_type = input['bias_type']
                counts[bias_type] += 1
            else:
                #sense = input['pos']
                sense = f'{input["pos"]} {input["sense"]}'
                counts[sense] += 1

            pro_sentence = input['stereotype']
            anti_sentence = input['anti-stereotype']
            print('pro_sentence: ', pro_sentence, 'anti_sentence: ', anti_sentence)
            if args.model == 'sense-base' or args.model == 'sense-large':
                pro_token_ids, pro_mask_ids = tokenizer(pro_sentence)
                anti_token_ids, anti_mask_ids = tokenizer(anti_sentence)
                pro_score = calculate_aul_for_sense_bert(model, pro_token_ids,
                                                         pro_mask_ids)
                anti_score = calculate_aul_for_sense_bert(model, anti_token_ids,
                                                          anti_mask_ids)
            elif args.model == 'bert-base' or args.model == 'bert-large':
                pro_token_ids = tokenizer.encode(pro_sentence, return_tensors='pt')
                anti_token_ids = tokenizer.encode(anti_sentence, return_tensors='pt')
                with torch.no_grad():
                    pro_score = calculate_aul_for_bert(model, pro_token_ids,
                                                       log_softmax)
                    anti_score = calculate_aul_for_bert(model, anti_token_ids,
                                                       log_softmax)
            print('pro_score: ', pro_score, 'anti_score: ', anti_score)
            total_score += 1
            if pro_score > anti_score:
                stereo_score += 1
                if args.data == 'cp' or args.data == 'ss':
                    scores[bias_type] += 1
                else:
                    scores[sense] += 1
            mlm_scores.append(pro_score)

    bias_score = round((stereo_score / total_score) * 100, 2)
    for bias_type, score in scores.items():
        bias_score = round((score / counts[bias_type]) * 100, 2)
        print(bias_type, bias_score)


if __name__ == "__main__":
    args = parse_args()
    main(args)
