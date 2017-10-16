from dataset import load_data
from dependency import *
from feature_extraction import get_features
from train import train_model
from utils import *
import pickle


def predict_arcs(conll_dict,model,feature_dict):
    buff = [i for i in range(len(conll_dict['FORM']))[::-1]]
    stack, dgraph = [], []
    while (len(buff) > 0 or len(stack) > 1):
        config = (stack,buff,dgraph)
        features = get_features(config,conll_dict)
        binary_features = one_hot_encoding([features],feature_dict)
        choice = model.predict(binary_features)
        try:
            if choice == 'shift':	shift(stack, buff, stack)
            elif choice == 'left_arc': left_arc(stack, buff, dgraph)
            elif choice == 'right_arc': right_arc(stack, buff, dgraph)
            else: return None
        except IndexError:
            break
    return dgraph


def test_model(file_name):
    test_data = flatten(load_data(file_name))
    with open('feature_dict.pkl', 'rb') as feature_f:
        feature_dict = pickle.load(feature_f)
    with open('model.pkl','rb') as model_f:
        model = pickle.load(model_f)

    all_predicted = []
    all_gold_arcs = []
    for id, conll_dict in test_data.items():
        gold_arcs = get_gold_arcs(conll_dict['HEAD'])
        projective = is_projective(gold_arcs, len(conll_dict['FORM']))
        if not projective or len(gold_arcs) == 1:
            continue
        all_gold_arcs.append(gold_arcs)
        predicted = predict_arcs(conll_dict,model,feature_dict)
        all_predicted.append(predicted)
    print_metrics(all_predicted,all_gold_arcs)


def print_metrics(predictions, gold_arcs):
    precision = []
    recall = []
    for prediction,gold_arc in zip(predictions, gold_arcs):
        correct_arcs = set(gold_arc) & set(prediction)
        if not prediction:
            precision.append(0)
        else:
            precision.append(float(len(correct_arcs))/len(prediction))
        recall.append(float(len(correct_arcs)) / len(gold_arc))
    total_precision = float(sum(precision))/float(len(precision))
    total_recall = float(sum(recall))/float(len(recall))
    f1 = 2*float(total_precision*total_recall)/float(total_precision+total_recall)

    print("Total precision :"+str(total_precision))
    print("Total recall: "+str(total_recall))
    print('F1 Measure: '+str(f1))

if __name__ == '__main__':
    train_model('en-ud-train.conllu')
    print("Training complete")
    test_model('en-ud-dev.conllu')
