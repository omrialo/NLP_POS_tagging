"""
intro2nlp, assignment 4, 2021

In this assignment you will implement a Hidden Markov model and an LSTM model
to predict the part of speech sequence for a given sentence.
(Adapted from Nathan Schneider)

"""

import torch
import torch.nn as nn
from torchtext.vocab import Vectors
from torchtext import data
import torch.optim as optim
from math import log, isfinite
from collections import Counter, OrderedDict
import numpy as np
import pandas as pd

import sys, os, time, platform, nltk, random

# With this line you don't need to worry about the HW  -- GPU or CPU
# GPU cuda cores will be used if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# You can call use_seed with other seeds or None (for complete randomization)
# but DO NOT change the default value.
def use_seed(seed = 1512021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_deterministic(True)
    #torch.backends.cudnn.deterministic = True

# utility functions to read the corpus
def who_am_i(): #this is not a class method
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name1': 'Omri Alon', 'id1': '302559539', 'email1': 'omrialo@post.bgu.ac.il',
            'name2': 'Anat Ben Haim', 'id2': '205467400', 'email2': 'anbe@post.bgu.ac.il'}


def read_annotated_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append( (word, tag) )
        line = f.readline()
    return sentence

def load_annotated_corpus(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_annotated_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_annotated_sentence(f)
    return sentences


START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"

allTagCounts = Counter()
perWordTagCounts = {}
transitionCounts = {}
emissionCounts = {}
A = {}  # transitions probabilities
B = {}  # emissions probabilities

def learn_params(tagged_sentences):
    """Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
     and emissionCounts data-structures.
    allTagCounts and perWordTagCounts are used for baseline tagging and
    do not include pseudo counts, dummy tags and unknowns.
    The transitionCounts and emissionCounts are computed with pseudo tags and are smoothed.
    A and B are the log-probability of the normalized counts, based on
    transitionCounts and  emissionCounts

    Args:
      tagged_sentences: a list of tagged sentences, each tagged sentence is a
       list of pairs (w,t), as retunred by load_annotated_corpus().

   Return:
      [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] (a list)
  """

    # Initialize variables
    global perWordTagCounts, emissionCounts, transitionCounts, allTagCounts, A, B
    emission = {}       # Holds list of words per tag
    transition = {}     # Holds list of tag 2 for tag 1
    perWord = {}        # Holds list of tags per word
    transition[START] = []

    for s in tagged_sentences:
        for i in range(len(s)):
            # For each word set the tag and word
            tag = s[i][1]
            word = s[i][0]

            # Add the tag count to the counter
            allTagCounts.update([tag])

            if word not in perWord.keys():
                # Initialize a list for the word if it's not in the dictionary
                perWord[word] = []
            # Add the tag to the list under the word
            perWord[word].append(tag)

            if tag not in emission.keys():
                # Initialize a list for the tag if it's not in the emission dictionary
                emission[tag] = []
            # Add the word to the list under the dictionary
            emission[tag].append(word)

            if i == 0:
                # If this is the beginning of the sentence, add it to the transition dict after the START token
                transition[START].append(tag)

            if i == len(s)-1:
                # If this is the end of the sentence, use the END token
                tOne = tag
                tTwo = END
            else:
                # Else use the next tag in the sentence
                tOne = tag
                tTwo = s[i+1][1]

            if tOne not in transition.keys():
                # Initialize a list for the tag if it's not in the transition dictionary
                transition[tOne] = []
            # Add the next tag to the list under the current tag
            transition[tOne].append(tTwo)

    # Turn dictionaries of lists into dictionaries of counters
    perWordTagCounts = {key: Counter(perWord[key]) for key in perWord.keys()}
    emissionCounts = {key: Counter(emission[key]) for key in emission.keys()}
    transitionCounts = {key: Counter(transition[key]) for key in transition.keys()}

    # Make sure all tags combinations are in the transition matrix
    # Turn counts into probabilities for matrix A
    tags_list1 = list(allTagCounts.keys()) + [START]
    tags_list2 = list(allTagCounts.keys()) + [END]
    for tag1 in tags_list1:
        A[tag1] = {}
        denom = (len(transitionCounts[tag1].keys()) + sum(transitionCounts[tag1].values()))
        for tag2 in tags_list2:
            if tag2 not in transitionCounts[tag1].keys():
                # If the combination doesn't exist, smooth the probability
                A[tag1][tag2] = log(1 / denom)
            else:
                # If it does exist, use the counts
                A[tag1][tag2] = log((1 + transitionCounts[tag1][tag2]) / denom)

    # Build matrix B with probabilities by emission counts and add the UNK token
    for key, val in list(emissionCounts.items()):
        denom = sum(val.values()) + len(val.keys())
        B[key] = {k: log((1 + val[k]) / denom) for k in val.keys()}
        B[key][UNK] = log(1 / denom)

    return [allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B]


def baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Each word is tagged by the tag most
    frequently associated with it. OOV words are tagged by sampling from the
    distribution of all tags.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        Return:
        list: list of pairs
    """
    tagged_sentence = []
    for i in range(len(sentence)):
        # Get word i
        word = sentence[i]
        if word in perWordTagCounts.keys():
            # If the word is known, take the most frequent tag for it
            tag = max(perWordTagCounts[word], key=perWordTagCounts[word].get)
        else:
            # If it's unknown, randomize a tag
            tag = random.choices(list(allTagCounts.keys()), weights=list(allTagCounts.values()), k=1)

        # Add the tagged word
        tagged_sentence.append((word, tag))

    return tagged_sentence
#
# #===========================================
# #       POS tagging with HMM
# #===========================================

class State:
    """The state class implements the Viterbi algorithm by including:

        word: the observation
        tag: the hidden state
        p: the best probability found for this state
        previous: a pointer to the previous state with the most probable path for this state
    """
    def __init__(self, word, tag, previous, p=None):
        self.word = word
        self.tag = tag
        self.previous = previous
        self.p = p

    def set_prob(self):
        """ A helper function that can set the new state best probability
        """

        seq = self.get_seq()
        prob = joint_prob(seq, A, B)
        self.p = prob

    def get_seq(self):
        """ This function return the sequence of states representations that leads to the best probability
            for the current state.

        return: list of tuples (word, tag), representing the states
        """

        if self.tag == START:
            return [(self.word, self.tag)]
        else:
            return self.previous.get_seq() + [(self.word, self.tag)]


def hmm_tag_sentence(sentence, A, B):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Tagging is done with the Viterbi
    algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): the HMM emission probabilities.

    Return:
        list: list of pairs
    """

    # Activate the Viterbi algorithm
    end_item = viterbi(sentence, A, B)
    # Get the full tagged sentence
    tagged_full = retrace(end_item)
    # Slice to remove dummy tokens, leaving only the original sentence
    tagged_sentence = tagged_full[1:-1]

    return tagged_sentence


def viterbi(sentence, A, B):
    """Creates the Viterbi matrix, column by column. Each column is a list of
        state object representing cells. Each cell ("item") is a state object
        that contains the word, tag, max probability and the previous state
        that led to this probability.

        The function returns the END item, from which it is possible to
        trace back to the beginning of the sentence.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): the HMM emission probabilities.

    Return:
        obj: the last item, tagged with END.

        """

    v_matrix = []  # Holds all possible states
    sent_s = State(None, START, None, 0)  # Create the start state

    for i in range(len(sentence)):
        word = sentence[i]
        if word in perWordTagCounts.keys():
            # If the word is known consider only its related tags
            tags = list(perWordTagCounts[word].keys())
        else:
            # Else consider all tags
            tags = list(allTagCounts.keys())

        v_matrix.append([])  # Create new matrix column
        for tag in tags:
            #  For each of the possible tags, get a sequence of previous tags
            if i == 0:
                previous = [sent_s]
            else:
                previous = v_matrix[i-1]

            # Get the best state (including path) for the tag and append to the column
            item = predict_next_best(word, tag, previous, A, B)
            v_matrix[i].append(item)

    # Get the best end state with the path that leads to it
    v_last = predict_next_best(None, END, v_matrix[-1], A, B)

    return v_last


def retrace(state):
    """Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). The list  corresponds to the
        list of words in the sentence (same indices).

    Args:
        state (state object): the last item returned by the Viterbi algorithm

    Return:
        list: list of pairs representing the sequence of states found by the Viterbi algorithm
    """

    if state.tag == START:
        # If reached root, return its state
        return [(state.word, state.tag)]
    else:
        # Add the current state to the previous states list
        return retrace(state.previous) + [(state.word, state.tag)]


def predict_next_best(word, tag, predecessor_list, A, B):
    """Returns a new item (state object) representing the best
        version for this cell in the matrix.
    """
    # Initialize probability the best path
    best_prob = 0
    best_item = None

    # Get the emission probability
    if word is None:
        # For dummy tokens
        p_emission = 0
    elif word not in perWordTagCounts.keys():
        # For an unknown word
        p_emission = B[tag][UNK]
    else:
        p_emission = B[tag][word]

    for previous in predecessor_list:
        # For each possible previous state calculate the probability
        # Using transition, emission and previous path probability
        prob = A[previous.tag][tag] + p_emission + previous.p

        if best_item is None or prob > best_prob:
            # Update if the new probability is the best so far
            best_prob = prob
            best_item = previous

    # Create the new state in the best way found
    new_state = State(word, tag, best_item, best_prob)

    return new_state


def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under
     the HMM model.

     Args:
         sentence (pair): a sequence of pairs (w,t) to compute.
         A (dict): The HMM Transition probabilities
         B (dict): the HMM emission probabilities.
     """
    p = 0   # Initialize the probability
    for i in range(len(sentence)):
        # Get word and tag i
        word = sentence[i][0]
        tag = sentence[i][1]

        # If this is the beginning of the sentence, consider the START tag
        # Else take the previous tag
        if i == 0:
            previous = START
        else:
            previous = sentence[i - 1][1]

        # Get the log probabilities and add to the joint probability
        A_prob = A[previous][tag]
        B_prob = B[tag][word]
        p += A_prob + B_prob

    # Add the END tag probability
    p += A[tag][END]

    assert isfinite(p) and p < 0  # Probability is not 0 or 1
    return p


#===========================================
#       POS tagging with BiLSTM
#===========================================

class BiLSTMPOSTagger(nn.Module):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 output_dim,
                 pad_idx,
                 hidden_dim=128,
                 n_layers=2,
                 bidirectional=True,
                 dropout=0.25):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout if n_layers > 1 else 0)

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]
        # pass text through embedding layer
        embedded = self.dropout(self.embedding(text))
        # embedded = [sent len, batch size, emb dim]
        # pass embeddings into LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs holds the backward and forward hidden states in the final layer
        # hidden and cell are the backward and forward hidden and cell states at the final time-step
        # output = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]
        # we use our outputs to make a prediction of what the tag should be
        predictions = self.fc(self.dropout(outputs))
        # predictions = [sent len, batch size, output dim]
        return predictions


class CaseBiLSTMPOSTagger(nn.Module):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 output_dim,
                 pad_idx,
                 n_layers=2,
                 hidden_dim=128,
                 bidirectional=True,
                 dropout=0.25
                 ):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.W2_case_layer_embedding = torch.tensor([[0,0,0],[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=torch.float32)#nn.Embedding(5, 3, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim+3,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout if n_layers > 1 else 0)

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, case_features):
        # text = [sent len, batch size]

        # pass text through embedding layer
        embedded = self.dropout(self.embedding(text))
        # embedded,word_features = [sent len, batch size, emb dim]
        word_features = self.dropout(self.W2_case_layer_embedding[case_features])
        #embeddedANDcase = embedded.add(word_features)
        embeddedANDcase = torch.cat((word_features, embedded), dim=2)
        # pass embeddings into LSTM
        outputs, (hidden, cell) = self.lstm(embeddedANDcase)

        # outputs holds the backward and forward hidden states in the final layer
        # hidden and cell are the backward and forward hidden and cell states at the final time-step

        # output = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]

        # we use our outputs to make a prediction of what the tag should be
        predictions = self.fc(self.dropout(outputs))
        # predictions = [sent len, batch size, output dim]
        return predictions

def normalize_text(text):
    """Returns a normalized string based on the specified string.
       Args:
           text (str): the text to normalize
       Returns:
           string. the normalized text.
    """
    return text.lower()

""" You are required to support two types of bi-LSTM:
    1. a vanilla biLSTM in which the input layer is based on simple word embeddings
    2. a case-based BiLSTM in which input vectors combine a 3-dim binary vector
        encoding case information, see
        https://arxiv.org/pdf/1510.06168.pdf
"""

def get_case_info(word):
    full_lowercase = 'full_lowercase'
    full_uppercase = 'full_uppercase'
    leading_capital_letter = 'leading_capital_letter'
    # full_lowercase = "1"
    # full_uppercase = "2"
    # leading_capital_letter = "3"
    if len(word) > 1:
        if word[0].isupper() and word[1].islower():
            return leading_capital_letter
    elif word.isupper():
        return full_uppercase
    return full_lowercase


def get_valid_data_caseLSTM(val_data):
    """Returns few objects needed in the "model" dictionary. the train data will be processed to fit for case lstm
       Args:
           val_data (list): in the same format as mentioned in "initialize_rnn"

       Returns:
           valid_data: validation data in an appropriate format
    """
    flattenTagList = [item[1] for items in val_data for item in items]
    unique_tags_list = set(flattenTagList)
    # tag_to_ix = {k: v for v, k in enumerate(unique_tags_list)}
    df = pd.DataFrame(
        {'sentence': [[0] for i in range(len(val_data))], 'case_features': [[0] for i in range(len(val_data))],
         'tags': [[18] for i in range(len(val_data))]},
        dtype='float')
    tag_seq = []
    words_seq = []
    case_features_seq = []
    index_row = 0
    index_col = 0
    for sentence in val_data:
        for word_tag in sentence:
            try:
                words_seq.append(normalize_text(word_tag[0]))
                case_features_seq.append(get_case_info(word_tag[0]))
                tag_seq.append(word_tag[1])
                index_col += 1
            except:
                tag_seq.append(word_tag[1])
                index_col += 1
        df['tags'][index_row] = [w for w in tag_seq]  # (ex4.prepare_tag_sequence(tag_seq, tag_to_ix))
        df['sentence'][index_row] = [w for w in words_seq]
        df['case_features'][index_row] = [w for w in case_features_seq]
        tag_seq.clear()
        case_features_seq.clear()
        words_seq.clear()
        index_col = 0
        index_row += 1

    #  setting the seed :
    use_seed()
    TEXT = data.Field(lower=True)
    TAGS = data.Field(unk_token=None)
    CASE_FEATURES = data.Field()
    fields = (("text", TEXT), ("case_features", CASE_FEATURES), ("tags", TAGS))
    val_data = [data.Example.fromlist([r.sentence, r.case_features, r.tags], fields) for r in df.itertuples()]

    return val_data


def get_valid_data(val_data):
    """Returns few objects needed in the "model" dictionary. the train data will be processed to fit for vanilla lstm
       Args:
           val_data (list): in the same format as mentioned in "initialize_rnn"

       Returns:
           valid_data: validation data in an appropriate format
    """
    flattenTagList = [item[1] for items in val_data for item in items]
    unique_tags_list = set(flattenTagList)
    # tag_to_ix = {k: v for v, k in enumerate(unique_tags_list)}
    df = pd.DataFrame(
        {'sentence': [[0] for i in range(len(val_data))], 'tags': [[18] for i in range(len(val_data))]},
        dtype='float')
    tag_seq = []
    words_seq = []
    index_row = 0
    index_col = 0
    for sentence in val_data:
        for word_tag in sentence:
            try:
                words_seq.append(normalize_text(word_tag[0]))
                tag_seq.append(word_tag[1])
                index_col += 1
            except:
                tag_seq.append(word_tag[1])
                index_col += 1
        df['tags'][index_row] = [w for w in tag_seq]  # (ex4.prepare_tag_sequence(tag_seq, tag_to_ix))
        df['sentence'][index_row] = [w for w in words_seq]
        tag_seq.clear()
        words_seq.clear()
        index_col = 0
        index_row += 1

    #  setting the seed :
    use_seed()
    TEXT = data.Field(lower=True)
    TAGS = data.Field(unk_token=None)
    fields = (("text", TEXT), ("tags", TAGS))
    val_data = [data.Example.fromlist([r.sentence, r.tags], fields) for r in df.itertuples()]

    return val_data

def preprocess_data_for_CaseLSTM(train_data, vectors_fn,MIN_FREQ=1, max_vocab_size=-1, BATCH_SIZE=16, val_data = None):
    """Returns few objects needed in the "model" dictionary. the train data will be processed to fit for case lstm
       Args:
           train_data(list): as mentioned in "initialize_rnn"
            vectors_fn(str): as mentioned in "initialize_rnn"
            MIN_FREQ (int): as mentioned in "initialize_rnn"
             max_vocab_size(int): as mentioned in "initialize_rnn"
              BATCH_SIZE(int): size of the batches
               val_data (list): as mentioned in "initialize_rnn"
       Returns:
           train_iterator, valid_iterator, PAD_IDX, TAG_PAD_IDX, TEXT, TAGS, INPUT_DIM, OUTPUT_DIM, TEXT.vocab
    """
    global embeddings_fn
    global min_frequency
    embeddings_fn = vectors_fn
    min_frequency = MIN_FREQ
    flattenTagList = [item[1] for items in train_data for item in items]
    unique_tags_list = set(flattenTagList)
    # tag_to_ix = {k: v for v, k in enumerate(unique_tags_list)}
    df = pd.DataFrame(
        {'sentence': [[0] for i in range(len(train_data))],'case_features': [[0] for i in range(len(train_data))],'tags': [[18] for i in range(len(train_data))]},
        dtype='float')
    tag_seq = []
    words_seq = []
    case_features_seq = []
    index_row = 0
    index_col = 0
    for sentence in train_data:
        for word_tag in sentence:
            try:
                words_seq.append(normalize_text(word_tag[0]))
                case_features_seq.append(get_case_info(word_tag[0]))
                tag_seq.append(word_tag[1])
                index_col += 1
            except:
                tag_seq.append(word_tag[1])
                index_col += 1
        df['tags'][index_row] = [w for w in tag_seq]  # (ex4.prepare_tag_sequence(tag_seq, tag_to_ix))
        df['sentence'][index_row] = [w for w in words_seq]
        df['case_features'][index_row] = [w for w in case_features_seq]
        tag_seq.clear()
        case_features_seq.clear()
        words_seq.clear()
        index_col = 0
        index_row += 1

    #  setting the seed :
    use_seed()
    TEXT = data.Field(lower=True)
    TAGS = data.Field(unk_token=None)
    CASE_FEATURES = data.Field()
    fields = (("text", TEXT), ("case_features", CASE_FEATURES), ("tags", TAGS))
    train_examp = [data.Example.fromlist([r.sentence, r.case_features, r.tags], fields) for r in df.itertuples()]

    if val_data is None:
        train_data = data.Dataset(train_examp, fields=fields)
        train, valid_data = data.Dataset.split(train_data, 0.9)
    else:
        train_data = data.Dataset(train_examp, fields=fields)
        valid_data = get_valid_data_caseLSTM(val_data)
        valid_data = data.Dataset(valid_data, fields=fields)

    if max_vocab_size == -1:
        max_vocab_size = None
    TEXT.build_vocab(train_data,
                     min_freq=MIN_FREQ,
                     max_size=max_vocab_size,
                     vectors=load_pretrained_embeddings(vectors_fn),
                     unk_init=torch.Tensor.normal_)
    TAGS.build_vocab(train_data)
    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_data, valid_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.text),
        device=device)
    CASE_FEATURES.build_vocab(train_data)
    INPUT_DIM = len(TEXT.vocab)
    # EMBEDDING_DIM = 100
    # HIDDEN_DIM = 128
    OUTPUT_DIM = len(TAGS.vocab)
    # N_LAYERS = 2
    # BIDIRECTIONAL = True
    # DROPOUT = 0.25
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    #
    # params_d = {'input_dimension': INPUT_DIM,
    #             'embedding_dimension': EMBEDDING_DIM,
    #             'num_of_layers': N_LAYERS,
    #             'output_dimension': OUTPUT_DIM}
    # model = BiLSTMPOSTagger(params_d)

    TAG_PAD_IDX = TAGS.vocab.stoi[TAGS.pad_token]

    return train_iterator, valid_iterator, PAD_IDX, TAG_PAD_IDX, TEXT, TAGS,CASE_FEATURES, INPUT_DIM, OUTPUT_DIM, TEXT.vocab

def preprocess_data_for_LSTM(train_data, vectors_fn,MIN_FREQ=1, max_vocab_size=-1, BATCH_SIZE=16, val_data = None):
    """Returns few objects needed in the "model" dictionary. the train data will be processed to fit for vanilla lstm
       Args:
           train_data(list): as mentioned in "initialize_rnn"
            vectors_fn(str): as mentioned in "initialize_rnn"
            MIN_FREQ (int): as mentioned in "initialize_rnn"
             max_vocab_size(int): as mentioned in "initialize_rnn"
              BATCH_SIZE(int): size of the batches
               val_data (list): as mentioned in "initialize_rnn"
       Returns:
           train_iterator, valid_iterator, PAD_IDX, TAG_PAD_IDX, TEXT, TAGS, INPUT_DIM, OUTPUT_DIM, TEXT.vocab
    """
    global embeddings_fn
    global min_frequency
    embeddings_fn = vectors_fn
    min_frequency = MIN_FREQ
    flattenTagList = [item[1] for items in train_data for item in items]
    unique_tags_list = set(flattenTagList)
    # tag_to_ix = {k: v for v, k in enumerate(unique_tags_list)}
    df = pd.DataFrame(
        {'sentence': [[0] for i in range(len(train_data))], 'tags': [[18] for i in range(len(train_data))]},
        dtype='float')
    tag_seq = []
    words_seq = []
    index_row = 0
    index_col = 0
    for sentence in train_data:
        for word_tag in sentence:
            try:
                words_seq.append(normalize_text(word_tag[0]))
                tag_seq.append(word_tag[1])
                index_col += 1
            except:
                tag_seq.append(word_tag[1])
                index_col += 1
        df['tags'][index_row] = [w for w in tag_seq]  # (ex4.prepare_tag_sequence(tag_seq, tag_to_ix))
        df['sentence'][index_row] = [w for w in words_seq]
        tag_seq.clear()
        words_seq.clear()
        index_col = 0
        index_row += 1

    #  setting the seed :
    use_seed()
    TEXT = data.Field(lower=True)
    TAGS = data.Field(unk_token=None)
    fields = (("text", TEXT), ("tags", TAGS))
    train_examp = [data.Example.fromlist([r.sentence, r.tags], fields) for r in df.itertuples()]

    if val_data is None:
        train_data = data.Dataset(train_examp, fields=fields)
        train, valid_data = data.Dataset.split(train_data, 0.9)
    else:
        train_data = data.Dataset(train_examp, fields=fields)
        valid_data = get_valid_data(val_data)
        valid_data = data.Dataset(valid_data, fields=fields)


    if max_vocab_size == -1:
        max_vocab_size = None
    TEXT.build_vocab(train_data,
                     min_freq=MIN_FREQ,
                     max_size=max_vocab_size,
                     vectors=load_pretrained_embeddings(vectors_fn),
                     unk_init=torch.Tensor.normal_)
    TAGS.build_vocab(train_data)
    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_data, valid_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.text),
        device=device)

    INPUT_DIM = len(TEXT.vocab)
    # EMBEDDING_DIM = 100
    # HIDDEN_DIM = 128
    OUTPUT_DIM = len(TAGS.vocab)
    # N_LAYERS = 2
    # BIDIRECTIONAL = True
    # DROPOUT = 0.25
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    #
    # params_d = {'input_dimension': INPUT_DIM,
    #             'embedding_dimension': EMBEDDING_DIM,
    #             'num_of_layers': N_LAYERS,
    #             'output_dimension': OUTPUT_DIM}
    # model = BiLSTMPOSTagger(params_d)

    TAG_PAD_IDX = TAGS.vocab.stoi[TAGS.pad_token]

    return train_iterator, valid_iterator, PAD_IDX, TAG_PAD_IDX, TEXT, TAGS, INPUT_DIM, OUTPUT_DIM, TEXT.vocab


def get_vocab(data_train, max_vocab_size=-1, min_frequency=1):
    """process vocabulary learned from data set

    Args:
        data_train (list): list of (w, t) tuples
        max_vocab_size (int): max vocabulary size
        min_frequency (int): the occurrence threshold to consider
        lower (bool): whether to lower case word in vocabulary or not

    Return
        vocab (list)
    """
    counter = Counter([normalize_text(word) for sentence in data_train for word, tag in sentence])
    counter = Counter(dict([(x, y) for x, y in counter.items() if y >= min_frequency]))
    if max_vocab_size > -1:
        counter = Counter(dict(counter.most_common(max_vocab_size)))
    vocab = list(set(counter.keys()))

    return vocab


def initialize_rnn_model(params_d):
    """Returns a dictionary with the objects and parameters needed to run/train_rnn
       the lstm model. The LSTM is initialized based on the specified parameters.
       thr returned dict is may have other or additional fields.

    Args:
        params_d (dict): a dictionary of parameters specifying the model. The dict
                        should include (at least) the following keys:
                        {'max_vocab_size': max vocabulary size (int),
                        'min_frequency': the occurence threshold to consider (int),
                        'input_rep': 0 for the vanilla and 1 for the case-base (int),
                        'embedding_dimension': embedding vectors size (int),
                        'num_of_layers': number of layers (int),
                        'output_dimension': number of tags in tagset (int),
                        'pretrained_embeddings_fn': str,
                        'data_fn': str
                        }
                        max_vocab_size sets a constraints on the vocab dimention.
                            If the its value is smaller than the number of unique
                            tokens in data_fn, the words to consider are the most
                            frequent words. If max_vocab_size = -1, all words
                            occuring more that min_frequency are considered.
                        min_frequency privides a threshold under which words are
                            not considered at all. (If min_frequency=1 all words
                            up to max_vocab_size are considered;
                            If min_frequency=3, we only consider words that appear
                            at least three times.)
                        input_rep (int): sets the input representation. Values:
                            0 (vanilla), 1 (case-base);
                            <other int>: other models, if you are playful
                        The dictionary can include other keys, if you use them,
                             BUT you shouldn't assume they will be specified by
                             the user, so you should spacify default values.
    Return:
        a dictionary with the at least the following key-value pairs:
                                       {'lstm': torch.nn.Module object,
                                       input_rep: [0|1]}
        #Hint: you may consider adding the embeddings and the vocabulary
        #to the returned dict
    """
    use_seed()
    global embedding_dimension
    data_train = load_annotated_corpus(params_d["data_fn"])
    max_vocab_size = params_d["max_vocab_size"]
    min_frequency = params_d["min_frequency"]
    embedding_dimension = params_d["embedding_dimension"]
    # vocab = get_vocab(data_train, max_vocab_size=max_vocab_size, min_frequency=min_frequency)
    # vectors = load_pretrained_embeddings_ref(params_d["pretrained_embeddings_fn"], vocab=vocab)

    # vectors = torch.FloatTensor(list(embedding.values()))
    # word_to_idx = dict(zip(embedding.keys(), range(len(embedding))))
    # hidden_dim = params_d["hidden_dim"] if "hidden_dim" in params_d else 10


    if params_d["input_rep"] == 0:  # Vanilla
        train_iterator, valid_iterator, PAD_IDX, TAG_PAD_IDX, TEXT, TAGS, INPUT_DIM, OUTPUT_DIM, vocab2 = preprocess_data_for_LSTM(
            data_train, params_d["pretrained_embeddings_fn"], min_frequency, max_vocab_size)
        input_dim = INPUT_DIM
        embedding_dim = embedding_dimension
        output_dim = OUTPUT_DIM  # params_d["output_dimension"] if "output_dimension" in params_d else OUTPUT_DIM
        pad_idx = PAD_IDX
        n_layers = params_d["num_of_layers"] if "num_of_layers" in params_d else 2
        model_object = BiLSTMPOSTagger(input_dim=input_dim,
                 embedding_dim=embedding_dim,
                 output_dim=output_dim,
                 pad_idx=pad_idx,
                 n_layers=n_layers).to(device)
    else:  # case-base
        train_iterator, valid_iterator, PAD_IDX, TAG_PAD_IDX, TEXT, TAGS, CASE_FEATURES, INPUT_DIM, OUTPUT_DIM, vocab2 = preprocess_data_for_CaseLSTM(
            data_train, params_d["pretrained_embeddings_fn"], min_frequency, max_vocab_size)
        input_dim = INPUT_DIM
        embedding_dim = embedding_dimension
        output_dim = OUTPUT_DIM  # params_d["output_dimension"] if "output_dimension" in params_d else OUTPUT_DIM
        pad_idx = PAD_IDX
        n_layers = params_d["num_of_layers"] if "num_of_layers" in params_d else 2
        model_object = CaseBiLSTMPOSTagger(input_dim=input_dim,
                 embedding_dim=embedding_dim,
                 output_dim=output_dim,
                 pad_idx=pad_idx,
                 n_layers=n_layers).to(device)

    model={}
    model['train_iterator'] = train_iterator
    model['valid_iterator'] = valid_iterator
    model['lstm'] = model_object
    model['TAG_PAD_IDX'] = TAG_PAD_IDX
    model['TEXT'] = TEXT
    model['TAGS'] = TAGS
    model['input_rep'] = params_d["input_rep"]

    return model


def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])


def load_pretrained_embeddings(path, vocab=None):
    """ Returns an object with the the pretrained vectors, loaded from the
        file at the specified path. The file format is the same as
        https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
        You can also access the vectors at:
         https://www.dropbox.com/s/qxak38ybjom696y/glove.6B.100d.txt?dl=0
         (for efficiency (time and memory) - load only the vectors you need)
        The format of the vectors object is not specified as it will be used
        internaly in your code, so you can use the datastructure of your choice.

    Args:
        path (str): full path to the embeddings file
        vocab (list): a list of words to have embeddings for. Defaults to None.

    """
    # IMPORTANT NOTE! : we do take in considerations the vocab, so we dont load the vecotrs we dont need:
    # we do that in the function "preprocess_data_for_LSTM"
    vocab = ['the','hi','.']
    vectors = Vectors(name=path)
    if vocab is not None:
        vectors.itos = vocab
        vectors = Vectors(name=path)
    return vectors



def train_epoch(model, iterator, optimizer, criterion, tag_pad_idx, input_rep):
    """used in the train_lstm loop. this function trains one epoch given train iterator and the model

    Return
        epoch training loss, epoch training accuracy
    """
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        text = batch.text
        tags = batch.tags

        optimizer.zero_grad()

        # text = [sent len, batch size]
        if input_rep == 1:
            case_features = batch.case_features
            predictions = model(text, case_features)
        else:
            predictions = model(text)

        # predictions = [sent len, batch size, output dim]
        # tags = [sent len, batch size]

        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)

        # predictions = [sent len * batch size, output dim]
        # tags = [sent len * batch size]

        loss = criterion(predictions, tags)

        acc = categorical_accuracy(predictions, tags, tag_pad_idx)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate_epoch(model, iterator, criterion, tag_pad_idx, input_rep):
    """used in the train_lstm loop. this function trains one epoch given validation iterator and the model

    Return
        epoch validation loss, epoch validation accuracy
    """
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text = batch.text
            tags = batch.tags

            if input_rep == 1:
                case_features = batch.case_features
                predictions = model(text, case_features)
            else:
                predictions = model(text)

            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)

            loss = criterion(predictions, tags)

            acc = categorical_accuracy(predictions, tags, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    """calculates elapsed_minutes and elapsed_seconds, given start and end time
    this function is used for the train lstm loop

    Return
        elapsed_mins, elapsed_secs
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def init_weights(m):
    """this function initializes the weights of a model with mean=0, std=0.1
    """
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)


def reportAccuracyOnDevSet(model):
    print ("reporting overall Accuracy on DEV data")
    dev_data = load_annotated_corpus('en-ud-dev.upos.tsv')
    sum_corrects = 0
    sum_words = 0
    for example in dev_data:
        sentence = [pair[0] for pair in example]
        pred_sentence = rnn_tag_sentence(sentence, model)
        gold_sentence = example
        correct, correctOOV, OOV = count_correct(gold_sentence, pred_sentence)
        sum_words += len(pred_sentence)
        sum_corrects += correct
        sum_corrects += correctOOV
    print("DEV data length ", str(len(dev_data)))
    print("total words tagged ", str(sum_words))
    print("ACCURACY ", str(sum_corrects/sum_words))

def train_rnn(model, train_data, val_data = None):
    """Trains the BiLSTM model on the specified data.

    Args:
        model (dict): the model dict as returned by initialize_rnn_model()
        train_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus()
        val_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus() to be used for validation.
                            Defaults to None
        input_rep (int): sets the input representation. Defaults to 0 (vanilla),
                         1: case-base; <other int>: other models, if you are playful
    """

    lstm = model["lstm"]
    input_rep = model["input_rep"]
    global TEXT
    global TAGS
    N_EPOCHS = 12
    lstm.apply(init_weights)
    if input_rep == 0:
        train_iterator, valid_iterator, PAD_IDX, TAG_PAD_IDX, TEXT, TAGS, INPUT_DIM, OUTPUT_DIM, vocab2 = preprocess_data_for_LSTM(train_data=train_data, vectors_fn=embeddings_fn, MIN_FREQ=min_frequency, val_data=val_data)
    else:
        global CASE_FEATURES
        train_iterator, valid_iterator, PAD_IDX, TAG_PAD_IDX, TEXT, TAGS, CASE_FEATURES, INPUT_DIM, OUTPUT_DIM, vocab2 = preprocess_data_for_CaseLSTM(train_data=train_data, vectors_fn=embeddings_fn, MIN_FREQ=min_frequency, val_data=val_data)

    global global_lstm_vocab
    pretrained_embeddings = TEXT.vocab.vectors
    # pretrained_case_embeddings = torch.tensor([[0,0,0],[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=torch.float32)

    print(pretrained_embeddings.shape)
    lstm.embedding.weight.data.copy_(pretrained_embeddings)
    lstm.embedding.weight.data[PAD_IDX] = torch.zeros(embedding_dimension)
    global_lstm_vocab = TEXT.vocab.itos
    criterion = nn.CrossEntropyLoss(ignore_index=TAG_PAD_IDX)
    optimizer = optim.Adam(lstm.parameters())
    lstm = lstm.to(device)

    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss, train_acc = train_epoch(lstm, train_iterator, optimizer, criterion, TAG_PAD_IDX, input_rep)
        valid_loss, valid_acc = evaluate_epoch(lstm, valid_iterator, criterion, TAG_PAD_IDX, input_rep)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(lstm.state_dict(), 'tut1-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
    # reportAccuracyOnDevSet({'lstm':lstm, 'input_rep':input_rep})
    sentence = "Hello, my name is Omri and I study in Ben Gurion university"
    rnn_tag_sentence(sentence, {'lstm':lstm, 'input_rep':input_rep})


def rnn_tag_sentence(sentence, model):
    """ Returns a list of pairs (w,t) where each w corresponds to a word
        (same index) in the input sentence and t is the predicted tag.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict):  a dictionary with the trained BiLSTM model and all that is needed
                        to tag a sentence.

    Return:
        list: list of pairs
    """
    lstm = model["lstm"]
    input_rep = model["input_rep"]
    lstm.eval()

    if isinstance(sentence, str):
        sentence = sentence.split(" ")
        tokens = [token for token in sentence]
        case_infos = [get_case_info(word) for word in sentence]
    else:
        tokens = [token for token in sentence]
        case_infos = [get_case_info(word) for word in sentence]


    tokens = [t.lower() for t in tokens]
    case_infos = [get_case_info(word) for word in sentence]

    numericalized_tokens = [TEXT.vocab.stoi[t] for t in tokens]
    unk_idx = TEXT.vocab.stoi[TEXT.unk_token]
    unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
    token_tensor = torch.LongTensor(numericalized_tokens)
    token_tensor = token_tensor.unsqueeze(-1).to(device)

    if input_rep == 1:
        numericalized_tokens_case = [CASE_FEATURES.vocab.stoi[t] for t in case_infos]
        unk_idx_case = CASE_FEATURES.vocab.stoi[CASE_FEATURES.unk_token]
        unks = [t for t, n in zip(case_infos, numericalized_tokens_case) if n == unk_idx_case]
        token_tensor_case = torch.LongTensor(numericalized_tokens_case)
        token_tensor_case = token_tensor_case.unsqueeze(-1).to(device)
        predictions = lstm(token_tensor, token_tensor_case)
    else:
        predictions = lstm(token_tensor)

    top_predictions = predictions.argmax(-1)

    predicted_tags = [TAGS.vocab.itos[t.item()] for t in top_predictions]
    tagged_sentence = []
    i = 0
    for w in sentence:
        tagged_sentence.append([w, predicted_tags[i]])
        i += 1

    return tagged_sentence


def get_best_performing_model_params():
    """Returns a disctionary specifying the parameters of your best performing
        BiLSTM model.
        IMPORTANT: this is a *hard coded* dictionary that will be used to create
        a model and train a model by calling
               initialize_rnn_model() and train_lstm()
    """
    UNIVERSAL_TAGS = ["UNK","ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
                      "PUNCT",
                      "SCONJ", "SYM", "VERB", "X"]
    model_params = \
        {'max_vocab_size': -1,
         'min_frequency': 1,
         'input_rep': 1,
         'embedding_dimension': 100,
         'num_of_layers': 2,
         'output_dimension': len(UNIVERSAL_TAGS),
         'pretrained_embeddings_fn': 'glove.6B.100d.txt',
         'data_fn': 'en-ud-train.upos.tsv'
         }
    return model_params


#===========================================================
#       Wrapper function (tagging with a specified model)
#===========================================================

def tag_sentence(sentence, model):
    """Returns a list of pairs (w,t) where pair corresponds to a word (same index) in
    the input sentence. Tagging is done with the specified model.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict): a dictionary where key is the model name and the value is
           an ordered list of the parameters of the trained model (baseline, HMM)
           or the model isteld and the input_rep flag (LSTMs).

        Models that must be supported (you can add more):
        1. baseline: {'baseline': [perWordTagCounts, allTagCounts]}
        2. HMM: {'hmm': [A,B]}
        3. Vanilla BiLSTM: {'blstm':[model_dict]}
        4. BiLSTM+case: {'cblstm': [model_dict]}
        5. (NOT REQUIRED: you can add other variations, agumenting the input
            with further subword information, with character-level word embedding etc.)

        The parameters for the baseline model are:
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        The parameters for the HMM are:
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

        Parameters for an LSTM: the model dictionary (allows tagging the given sentence)


    Return:
        list: list of pairs
    """
    key = list(model.keys())[0]
    values = model[key]

    if key == 'baseline':
        return baseline_tag_sentence(sentence, *values)
    if key == 'hmm':
        return hmm_tag_sentence(sentence, *values)
    if key == 'blstm':
        dict = {"lstm": values['lstm'], "input_rep": values['input_rep']}
        return rnn_tag_sentence(sentence, dict)
    if key == 'cblstm':
        dict = {"lstm": values['lstm'], "input_rep": values['input_rep']}
        return rnn_tag_sentence(sentence, dict)

def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags, the total number of
    correctly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger

    """
    assert len(gold_sentence) == len(pred_sentence)

    correct = 0
    correctOOV = 0
    OOV = 0
    words = set()

    # Get a comprehensive list of words in vocabulary
    if len(perWordTagCounts) > 0:
        words.update(list(perWordTagCounts.keys()))
    if 'global_lstm_vocab' in globals():
        words.update(global_lstm_vocab)

    for i in range(len(gold_sentence)):
        # For each word
        if gold_sentence[i][1] == pred_sentence[i][1]:
            # If tagged correctly
            correct += 1
            if gold_sentence[i][0] not in words:
                # If also OOV
                correctOOV += 1
                OOV += 1
        elif gold_sentence[i][0] not in words:
            # If not tagged correctly but is OOV
            OOV += 1

    return correct, correctOOV, OOV
