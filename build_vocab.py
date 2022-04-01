import re
import os
import json
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle

data_dir = '/scratch/alpine/mecr8410/deeplearning/lab4/data/'


def remove_polite(words):
    polite_words = {'hello there', 'hello', 'please', 'i was wondering', 'would you mind', 'i need help',  'can you tell me',
                    'good morning', 'can you help me', 'could you tell me', 'if possible', 'good evening', 'have a good day',
                    'hi there', 'thanks so much', 'thank you so much', 'thanks', 'thank you',  'goodbye', 'have a great day', 'good day'}
    for rm_word in polite_words:
        words = words.replace(rm_word, '')
    return words

def get_data(data_dir, dataset):
    ds_fn = dataset+'.json'
    annotation_fn = os.path.join(data_dir, 'Annotations', ds_fn)
    if not os.path.exists(annotation_fn):
        print('{} does not exist'.format(annotation_file))
    with open(annotation_fn) as annot_data:
        data = json.load(annot_data)
    return data

def get_question_vocab(data):
    all_q_words = ''
    for annotation in data:
        question = annotation['question'].lower()
        all_q_words += ' ' + question
    all_q_words = remove_polite(all_q_words)
    all_q_words = re.sub(r'[^A-Za-z0-9 ]+', '', all_q_words)
    tokens = word_tokenize(all_q_words)
    #stop_words = set(stopwords.words('english'))
    #tokens = [w for w in tokens if not w in stop_words and w.isalpha()]
    frequency_dist = nltk.FreqDist(tokens)
    sort_tokens = sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)
    vocab = {word:idx+1 for idx, word in enumerate(sort_tokens)}

    return vocab

def get_answer_vocab(data):
    all_a_words = ''
    for annotation in data:
        if annotation['answer_type'] != 'unanswerable':
            for ans in annotation['answers']:
                answer = ans['answer'].lower()
                all_a_words += ' ' + answer.replace("'", "")
    all_a_words = remove_polite(all_a_words)
    tokens = word_tokenize(all_a_words)
    #stop_words = set(stopwords.words('english'))
    #tokens = [w for w in tokens if not w in stop_words and w.isalpha()]
    all_a_words = re.sub(r'[^A-Za-z0-9 ]+', '', all_a_words)
    frequency_dist = nltk.FreqDist(tokens)
    sort_tokens = sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)
    vocab = {word:idx+1 for idx, word in enumerate(sort_tokens)}
    return vocab

data = get_data(data_dir, 'train')
train_q_vocab = get_question_vocab(data)
train_a_vocab = get_answer_vocab(data)
vocab = {'question': train_q_vocab, 'answer': train_a_vocab}
pickle.dump(vocab, open('train_vocab.pickle', 'wb'))

data = get_data(data_dir, 'val')
train_q_vocab = get_question_vocab(data)
train_a_vocab = get_answer_vocab(data)
vocab = {'question': train_q_vocab, 'answer': train_a_vocab}
pickle.dump(vocab, open('val_vocab.pickle', 'wb'))
