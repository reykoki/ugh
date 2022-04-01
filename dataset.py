import pickle
from torch.utils.data import Dataset, DataLoader

class VizWizDataset(Dataset):
    def __init__(self, data_dict):
        self.encoded_questions = data_dict['encoded_questions']
        self.encoded_answers   = data_dict['encoded_answers']
        self.image_features    = data_dict['image_features']
        self.num_tokens        = data_dict['embed_quest_len']
        self.num_img_feats     = len(self.image_features[0])
        self.num_answers       = len(self.encoded_answers[0])
    def __len__(self):
        return len(self.encoded_questions)
    def __getitem__(self, idx):
        embed_quest = self.encoded_questions[idx]
        embed_ans = self.encoded_answers[idx]
        img_feats = self.image_features[idx]
        return embed_quest, embed_ans, img_feats

