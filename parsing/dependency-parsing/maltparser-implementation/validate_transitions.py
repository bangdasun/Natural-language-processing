from dependency import *
from dataset import load_data
from utils import get_gold_arcs
import random

# Run the file to check if transitions.py has been implemented correctly.
if __name__ == '__main__':
    valid_projective_sentences = ['Service does not get any better!!!!', 'Its now my home.',
                                  'Great graphic design work!', 'But it really does not matter.', 'simple but perfect',
                                  'what do you think?']

    random_sample = random.sample(valid_projective_sentences,1)[0]
    train_data = load_data('en-ud-train.conllu')

    sample_dict = {}
    for genre, sentence_group in train_data.items():
        for sentence_group_id, sentences_dict in sentence_group.items():
            for sentence_id, conll_dict in list(sentences_dict.items()):
                if conll_dict['TEXT'].strip() == random_sample:
                    sample_dict = conll_dict
                    break

    print("FORMS : %s"%(sample_dict['FORM']))

    gold_arcs = get_gold_arcs(sample_dict['HEAD'])

    print("GOLD ARCS : %s"%(gold_arcs))

    buff = [i for i in range(len(sample_dict['FORM']))[::-1]]

    dgraph, configs = make_transitions(buff, oracle_std, gold_arcs)

    print("TRANSITIONS : \n%s" %('\n'.join([config[2] for config in configs])))
