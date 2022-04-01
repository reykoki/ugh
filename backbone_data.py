import numpy as np
import torch
from pathlib import Path
import os
import json
import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
import pickle
import torchvision.transforms as transforms
os.environ['TORCH_HOME'] = './testing/'
import torchvision.models as models
from torch.autograd import Variable
from PIL import Image
import re

data_dir = '/scratch/alpine/mecr8410/deeplearning/lab4/data/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

max_len_question = 25

# Load the pretrained model
model = models.efficientnet_b0(pretrained=True)

# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

# Set model to evaluation mode
model.eval()

# Image transforms
scale = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

# https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c
def get_image_features(image_name):
    # load image
    img = Image.open(image_name)
    # transform
    t_img = Variable(normalize(to_tensor(scale(img))).unsqueeze(0))
    # my_embeddings will hold the feature vector
    # the 'avgpool' layer has an output size of 2048 for resnet152
    my_embedding = torch.zeros(1280)
    # store data
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(o.data.size(1)))
    # attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # run the model on transformed image
    model(t_img)
    # detach the copy function from the layer
    h.remove()
    # return the feature vector
    return my_embedding


def remove_polite(words):

    polite_words = {'hello there', 'hello', 'please', 'i was wondering', 'would you mind', 'i need help',  'can you tell me',
                    'good morning', 'can you help me', 'could you tell me', 'if possible', 'good evening', 'have a good day',
                    'hi there', 'thanks so much', 'thank you so much', 'thanks', 'thank you',  'goodbye', 'have a great day', 'good day'}
    for rm_word in polite_words:
        words = words.replace(rm_word, '')
    return words

def encode_question(question, question_vocab):
    question = question.lower()

    question = re.sub(r'[^A-Za-z0-9 ]+', '', question)
    question = remove_polite(question)
    tokens = word_tokenize(question)
    embed_q = torch.zeros(max_len_question, device=device).long()
    for i, token in enumerate(tokens):
        if i < max_len_question:
            idx = question_vocab.get(token, 0)
            embed_q[i] = idx
        else:
            break
    return embed_q

def encode_answers(answer_list, answers_vocab):
    embed_a = torch.zeros(len(answers_vocab), device=device).long()
    for answer_dict in answer_list:
        answer = answer_dict['answer'].lower()
        answer = re.sub(r'[^A-Za-z0-9 ]+', '', answer)
        answer = remove_polite(answer)
        tokens = word_tokenize(answer)
        # order doesn't matter for answers so just want to see which words come up
        for i, token in enumerate(tokens):
            idx = answers_vocab.get(token, 0)
            embed_a[idx] += 1
    return embed_a

def get_image(img_fn, data_dir, dataset):
    img_file = os.path.join(data_dir, dataset, img_fn)
    img_feats = get_image_features(img_file)
    return img_feats


def get_data_dict(data, vocab, num_annots):
    question_vocab = vocab['question']
    answers_vocab = vocab['answer']

    preprocessed_data = {'encoded_questions': [], 'encoded_answers': [], 'image_features': [], 'embed_quest_len': len(question_vocab), 'num_answers': len(answers_vocab)}
    #for annot in data[0:num_annots]:
    for idx, annot in enumerate(data):
        # remove unanswered questions
        #print(annot)
        if len(annot['answers']) > 0:
            print(idx)
            encoded_q = encode_question(annot['question'], question_vocab)
            encoded_a = encode_answers(annot['answers'], answers_vocab)
            img_feats = get_image(annot['image'], data_dir, dataset)
            preprocessed_data['encoded_questions'].append(encoded_q)
            preprocessed_data['encoded_answers'].append(encoded_a)
            preprocessed_data['image_features'].append(img_feats)
    return preprocessed_data

dataset = 'train'
ds_fn = dataset+'.json'
annotation_fn = os.path.join(data_dir, 'Annotations', ds_fn)
if not os.path.exists(annotation_fn):
    print('{} does not exist'.format(annotation_file))
with open(annotation_fn) as annot_data:
    train_data = json.load(annot_data)

train_vocab = pickle.load(open('train_vocab.pickle', 'rb'))
preprocessed_data = get_data_dict(train_data, train_vocab, 3000)
pickle.dump(preprocessed_data, open('bb_pre_proc_train_data.pickle', 'wb'))

dataset = 'val'
ds_fn = dataset+'.json'
annotation_fn = os.path.join(data_dir, 'Annotations', ds_fn)
if not os.path.exists(annotation_fn):
    print('{} does not exist'.format(annotation_file))
with open(annotation_fn) as annot_data:
    val_data = json.load(annot_data)

val_vocab = pickle.load(open('val_vocab.pickle', 'rb'))
preprocessed_data = get_data_dict(val_data, train_vocab, 500)
pickle.dump(preprocessed_data, open('bb_pre_proc_val_data.pickle', 'wb'))



