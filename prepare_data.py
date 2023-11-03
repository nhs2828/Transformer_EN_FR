import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import string, re
import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def remove_punctuations(text):
    for punctuation in string.punctuation.replace('<', '').replace('>', ''):
        text = re.sub(r'[^\x00-\x7F]', '',text.replace(punctuation, '')).replace('  ',' ') # remove punctuation -> remove unicode -> remove double space
    return text

def string2code(s, dictionary_word2id):
    s = s.split(' ')
    return torch.tensor([dictionary_word2id[c] for c in s])

def code2string(t, dictionary_id2word):
    if type(t) !=list:
        t = t.tolist()
    return ' '.join(dictionary_id2word[i] for i in t)

class en_fr_dataset(Dataset):
    def __init__(self, en, fr, max_len, word2id_en, word2id_fr):
        self.en = []
        self.fr = []
        self.word2id_en = word2id_en
        self.word2id_fr = word2id_fr
        self.max_len = max_len
        for sentence_en, sentence_fr in zip(en, fr):
            if len(sentence_en.split(' ')) <= max_len and len(sentence_fr.split(' ')) <= max_len:
                self.en.append(sentence_en)
                self.fr.append(sentence_fr)
        
    def __len__(self):
        return len(self.fr)
    
    def __getitem__(self, id):
        return string2code(self.en[id], self.word2id_en), string2code(self.fr[id], self.word2id_fr)
    
def pad_collate_fn(samples):
    maxlen = max([len(p[0]) for p in samples])
    data_x_encoder = torch.empty(size=(len(samples), maxlen))
    data_x_decoder = torch.empty(size=(len(samples), maxlen))
    data_y = torch.empty(size=(len(samples), maxlen))
    for i, (sentence_en, sentence_fr) in enumerate(samples):
        data_x_encoder[i] = F.pad(sentence_en, pad=(0,maxlen-len(sentence_en)), mode='constant', value=word2id_en['<pad>'])
        data_x_decoder[i] = F.pad(sentence_fr[:-1], pad=(0,maxlen-len(sentence_fr[:-1])), mode='constant', value=word2id_en['<pad>']) # shift right
        data_y[i] = F.pad(sentence_fr[1:], pad=(0,maxlen-len(sentence_fr[1:])), mode='constant', value=word2id_fr['<pad>'])
    data_x_encoder, data_x_decoder, data_y = data_x_encoder.type(torch.long), data_x_decoder.type(torch.long), data_y.type(torch.long)
    return (data_x_encoder, data_x_decoder), data_y

PATH = 'data/'
df = pd.read_csv(PATH+'eng_french.csv', encoding="utf-8")

df.columns = ['EN', 'FR']

df['EN'] = df['EN']+" <EOS>"
df['FR'] = "<BOS> "+df['FR']+" <EOS>"

punc = string.punctuation.replace('<', '').replace('>', '')
df['EN'] = df['EN'].str.lower().apply(remove_punctuations)
df['FR'] = df['FR'].str.lower().apply(remove_punctuations)
ds_en = df['EN'].values
ds_fr = df['FR'].values

cv_en = CountVectorizer(token_pattern='[a-z0-9<>]+')
cv_en.fit(ds_en)
vocab_en = cv_en.get_feature_names_out()
cv_fr = CountVectorizer(token_pattern='[a-z0-9<>]+')
cv_fr.fit(ds_fr)
vocab_fr = cv_fr.get_feature_names_out()
vocab_size_en = len(vocab_en)
vocab_size_fr = len(vocab_fr)

PAD_IX = 0
# English
id2word_en = dict(zip(range(1,len(vocab_en)+1),vocab_en))
id2word_en[PAD_IX] = '<pad>' # NULL CHARACTER
word2id_en = dict(zip(id2word_en.values(),id2word_en.keys()))
# French
id2word_fr = dict(zip(range(1,len(vocab_fr)+1),vocab_fr))
id2word_fr[PAD_IX] = '<pad>' # NULL CHARACTER
word2id_fr = dict(zip(id2word_fr.values(),id2word_fr.keys()))

final_vocab_size_en = len(word2id_en) # add pad
final_vocab_size_fr = len(word2id_fr) # add pad