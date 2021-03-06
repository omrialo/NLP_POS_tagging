import tagger_submitted

def calc_score(dev_data, model_dict):
    score_nom, score_denom = 0, 0
    correctOOV_nom = 0
    OOV_denom = 0
    for gold_sentence in dev_data:
        pred_sentence = [w[0] for w in gold_sentence]
        tagged_sentence = tagger_submitted.tag_sentence(pred_sentence, model_dict)
        correct, correctOOV, OOV = tagger_submitted.count_correct(gold_sentence, tagged_sentence)
        score_nom += (correct+correctOOV)
        correctOOV_nom += correctOOV
        OOV_denom += OOV
        score_denom += len(pred_sentence)
    print(f"{list(model_dict.keys())[0]} OOV score is {correctOOV_nom / OOV_denom}")
    print(f"{list(model_dict.keys())[0]} score is {score_nom / score_denom}")


def check_sampled_sentence(gold_sentence, model_dict):
    pred_sentence = [w[0] for w in gold_sentence]
    tagged_sentence = tagger_submitted.tag_sentence(pred_sentence, model_dict)
    correct, correctOOV, OOV = tagger_submitted.count_correct(gold_sentence, tagged_sentence)
    print("OOV accuracy:", str(correctOOV/OOV))
    print(f"correct: {correct}, correctOOV: {correctOOV}, OOV: {OOV}\n")


train_path = 'en-ud-train.upos.tsv'
dev_path = 'en-ud-dev.upos.tsv'

train_data = tagger_submitted.load_annotated_corpus(train_path)
dev_data = tagger_submitted.load_annotated_corpus(dev_path)
#
[allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] = tagger_submitted.learn_params(train_data)


#draw random sentnece
gold_sentence = dev_data#dev_data[randrange(len(dev_data))]
print(f"tested random sentence is {gold_sentence} of length {len(gold_sentence)}\n")


#test beseline
calc_score(dev_data, {'baseline': [perWordTagCounts, allTagCounts]})
# check_sampled_sentence(gold_sentence, {'baseline': [perWordTagCounts, allTagCounts]})


#test hmm
calc_score(dev_data, {'hmm': [A,B]})
# check_sampled_sentence(gold_sentence, {'hmm': [A,B]})


#LSTM settings:
pretrained_embeddings_fn = "glove.6B.100d.txt" # r"C:\src\MastersCourses\NLP\Assign_4\embed\glove.6B.100d.txt"
# model_dict = {'max_vocab_size': -1, #max vocabulary size(int),
#                 'min_frequency': 1, #the occurence threshold to consider(int),
#                 'input_rep': 1,
#                 'embedding_dimension': 100, #embedding vectors size (int),
#                 'num_of_layers': 2, #number of layers (int),
#                 'output_dimension': 17,#len(set(B.keys())), #number of tags in tagset (int),
#                 'pretrained_embeddings_fn': pretrained_embeddings_fn, #str,
#                 'data_fn': train_path #str
#             }
model_dict = tagger_submitted.get_best_performing_model_params()


#test Vanilla BiLSTM:
print(f"provided model dict is {model_dict}")
print(f"initializing model")
model = tagger_submitted.initialize_rnn_model(model_dict)
print(f"training model")
tagger_submitted.train_rnn(model, train_data)
print(f"evaluating model")
calc_score(dev_data, {'blstm': model})
check_sampled_sentence(gold_sentence, {'blstm': model})


#test BiLSTM + case:
model_dict['input_rep'] = 0
print(f"provided model dict is {model_dict}")
print(f"initializing model")
model = tagger_submitted.initialize_rnn_model(model_dict)
print(f"training model")
tagger_submitted.train_rnn(model, train_data)
print(f"evaluating model")
calc_score(dev_data, {'cblstm': model})
check_sampled_sentence(gold_sentence, {'cblstm': model})


