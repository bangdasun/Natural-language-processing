import numpy as np

def get_gold_arcs(head_list):
    gold_arcs = []
    for i, val in enumerate(head_list):
        try :
            val_int = int(val)
            gold_arcs.append((i, val_int))
        except :
            continue
    return gold_arcs

# Converts data from multi layer dictionary form to single layer dictionary
def flatten_domain(data):
    flattened_data = {}
    for sentence_group_id,sentences_dict in data.items():
        for sentence_id, conll_dict in list(sentences_dict.items()):
            flattened_data[sentence_group_id + sentence_id] = conll_dict
    return flattened_data

def flatten(data):
    flattened_data = {}
    for genre,sentence_group in data.items():
        for sentence_group_id,sentences_dict in sentence_group.items():
            for sentence_id, conll_dict in list(sentences_dict.items()):
                flattened_data[genre + sentence_group_id + sentence_id] = conll_dict
    return flattened_data

def is_projective(arcs,total_len):
    for arc in arcs:
        start, end = arc if arc[0] < arc[1] else (arc[1],arc[0])
        for k in range(start+1,end):
            for m in list(range(0,arc[0]))+list(range(end+1,total_len+1)):
                if (k,m) in arcs or (m,k) in arcs:
                    return False
    return True

def one_hot_encoding(X,feature_dict):
    binary_features = []
    for row in X:
        row_features = [0]*len(feature_dict)
        for feature in row:
            if feature in feature_dict:
                row_features[feature_dict[feature]] = 1
        binary_features.append(row_features)
    return np.array(binary_features)

