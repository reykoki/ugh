import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn.utils.weight_norm import weight_norm


class VizWizNet(nn.Module):
    def __init__(self, dataset, embed_dim, num_hid):
        super(VizWizNet, self).__init__()
        self.w_emb      = WordEmbedding(dataset.num_tokens, embed_dim, 0.0)
        self.q_emb      = QuestionEmbedding(embed_dim, num_hid, 1, 0.0)
        self.v_att      = Attention(dataset.num_img_feats, self.q_emb.num_hid, num_hid)
        self.q_net      = FCNN(num_hid, num_hid)
        self.v_net      = FCNN(dataset.num_img_feats, num_hid)
        self.classifier = Classifier(num_hid, 2 * num_hid,
                                dataset.num_answers, 0.5)

    def forward(self, v, q):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)

        v_emb = (att * v)
        #v_emb = (att * v).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        # use logits instead of prob
        logits = self.classifier(joint_repr)
        return logits


class FCNN(nn.Sequential):
    def __init__(self, in_dim, out_dim):
        super(FCNN, self).__init__()
        self.layer1 = weight_norm(nn.Linear(in_dim, out_dim), dim=None)
        self.relu1  = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        return x


class Attention(nn.Module):
    def __init__(self, num_img_feats, ntokens, num_hid):
        super(Attention, self).__init__()
        #in_dim = num_img_feats + ntokens
        in_dim = num_img_feats*2
        out_dim = num_hid
        self.nonlinear = FCNN(in_dim, out_dim)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def logits(self, v, q):
        num_objs = v.size(1)
        #q = q.unsqueeze(1).repeat(1, num_objs, 1)
        q = q.repeat(1,2)
        vq = torch.cat((v, q), dim=1)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits

    def forward(self, v, q):
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

class Classifier(nn.Sequential):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.0):
        super(Classifier, self).__init__()
        self.drop1 = nn.Dropout(dropout)
        self.layer1 = weight_norm(nn.Linear(in_dim, hid_dim), dim=None)
        self.relu = nn.ReLU()
        self.drop2 = nn.Dropout(dropout, inplace=False)
        self.out_layer = weight_norm(nn.Linear(hid_dim, out_dim), dim=None)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.drop2(x)
        return self.out_layer(x)

class WordEmbedding(nn.Module):
    """Word Embedding
    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, ntoken, emb_dim, dropout):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        return emb


class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, dropout):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()

        self.rnn = nn.GRU(
            in_dim, num_hid, nlayers,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        # number of directions (bidrectional)
        self.num_dirs = 2

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers, batch, self.num_hid)
        return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)
