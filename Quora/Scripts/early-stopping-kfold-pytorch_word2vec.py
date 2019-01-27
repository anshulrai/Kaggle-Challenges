import gc
import re
import os
import pandas as pd
import numpy as np
import random
from sklearn import metrics
import string
import math
import operator
import time
from keras.preprocessing import text, sequence
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data
from gensim import utils

### TO DO: Don't ignore embedding < embed_size, remove newly added misspellings

change_string = 'Added lower case embeddings for words!\n'

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# GENERAL HYPERPARAMS
num_folds = 5
seed = 42

# HYPERPARAMS FOR TEXT PROCESSING
max_features = 120000
maxlen = 100

# HYPERPARAMS FOR NN
batch_size = 1024
epochs_fixed = 4
epochs_trainable = 1
embed_size = 300
early_stopping_patience = 2
hidden_size = 60

set_seed(seed)

# PATH TO DATA DIRECTORY
PATH = "../input/"

puncts = {',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√'}

def clean_text(x):
    x = str(x)
    table = str.maketrans({key: ' {punct} ' for key in puncts})
    return x.translate(table)

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x

mispell_dict = {
"tamilans" : "tamilians",
"coinbase" : "digital currency exchange",
"upwork" : "freelancing platform",
"sjws" : "social justice warriors",
"feku" : "liar",
"quorans" : "people who use quora",
"qoura": "quora",
"redmi" : "phone",
"gdpr" : "regulation",
"oneplus" : "phone",
"xiomi" : "phone",
"ipill" : "contraception", # same below this
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "i would",
"i'd" : "i had",
"i'll" : "i will",
"i'm" : "i am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "i have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" we will",
"didn't": "did not",
"tryin'":"trying"}

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

def replace_typical_misspell(text):
    mispellings, mispellings_re = _get_mispell(mispell_dict)
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)

def threshold_search(y_true, y_proba):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.001) 
    F = 2/(1/precision + 1/recall)
    best_score = np.max(F)
    best_th = thresholds[np.argmax(F)]
    search_result = {'threshold': best_th, 'f1': best_score}
    return search_result

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class GRUNet(nn.Module):
    def __init__(self, embedding_matrix):
        super(GRUNet, self).__init__()
        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        self.gru_1 = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru_2 = nn.GRU(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)
        
        self.gru_1_attention = Attention(hidden_size*2, maxlen)
        self.gru_2_attention = Attention(hidden_size*2, maxlen)
        
        self.linear = nn.Linear(hidden_size*8, 16)
        self.relu = nn.ReLU()

        self.out = nn.Linear(16, 1)
        
    def forward(self, x):
        
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        
        h_gru_1, _ = self.gru_1(h_embedding)
        h_gru_2, _ = self.gru_2(h_gru_1)
        
        h_gru_1_atten = self.gru_1_attention(h_gru_1)
        h_gru_2_atten = self.gru_2_attention(h_gru_2)
        
        avg_pool = torch.mean(h_gru_2, 1)
        max_pool, _ = torch.max(h_gru_2, 1)
        
        conc = torch.cat((h_gru_1_atten, h_gru_2_atten, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        out = self.out(conc)
        
        return out

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
# LOAD DATA
train_df = pd.read_csv(PATH+'train.csv', usecols=['question_text', 'target'])
test_df = pd.read_csv(PATH+'test.csv', usecols = ['question_text'])

# 3RD PARTY CLEAN
train_df["question_text"] = train_df["question_text"].apply(lambda x: x.lower())
test_df["question_text"] = test_df["question_text"].apply(lambda x: x.lower())

train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))
test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))

train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_numbers(x))
test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_numbers(x))

train_df["question_text"] = train_df["question_text"].apply(lambda x: replace_typical_misspell(x))
test_df["question_text"] = test_df["question_text"].apply(lambda x: replace_typical_misspell(x))

# FOR CREATING PROCESSED DATA AND LABELS
train_sentences = train_df['question_text']
train_labels = train_df['target']
test_sentences = test_df['question_text']

del train_df, test_df

gc.collect()

# TOKENIZE TEXT
tokenizer = text.Tokenizer(num_words=max_features, oov_token='OOV')
tokenizer.fit_on_texts(list(train_sentences) + list(test_sentences))

tokenized_train = tokenizer.texts_to_sequences(train_sentences)
X_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)

tokenized_test = tokenizer.texts_to_sequences(test_sentences)
X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)

del tokenized_test, tokenized_train, train_sentences, test_sentences
gc.collect()

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index)+1)

# LIST OF ALL EMBEDDINGS USED
embedding_list = [PATH+'embeddings/paragram_300_sl999/paragram_300_sl999.txt', 
PATH+'embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec',
PATH+'embeddings/glove.840B.300d/glove.840B.300d.txt']

# MEAN AND STD VALUES FOR EMBEDDINGS
emb_mean_dict = {'paragram_300_sl999':-0.005324783269315958,
            'wiki-news-300d-1M':-0.0033469984773546457,
            'glove.840B.300d':-0.005838498938828707,
            'GoogleNews-vectors-negative300':-0.0051106834}

emb_std_dict = {'paragram_300_sl999':0.4934646189212799,
            'wiki-news-300d-1M':0.10985549539327621,
            'glove.840B.300d':0.4878219664096832,
            'GoogleNews-vectors-negative300':0.18445626}

global_mean = np.mean([i for i in emb_mean_dict.values()])
global_std = np.mean([i for i in emb_std_dict.values()])
global_embedding = np.random.normal(global_mean, global_std, (nb_words, embed_size))
embedding_count = np.zeros((nb_words,1))

for EMBEDDING_FILE in embedding_list:
    embedding_name = EMBEDDING_FILE.split('/')[3]
    for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore'):
        word, vec = o.split(' ', 1)
        if word not in word_index:
            word = word.lower()
            if word not in word_index:
                continue
        i = word_index[word]
        if i >= nb_words:
            continue
        embedding_vector = np.asarray(vec.split(' '), dtype='float32')[:embed_size]
        if len(embedding_vector) == embed_size:
            if embedding_count[i] == 0:
                global_embedding[i] = embedding_vector
            else:
                global_embedding[i] = (embedding_count[i]*global_embedding[i] + embedding_vector)/(embedding_count[i] + 1)
            embedding_count[i] += 1
    del embedding_vector
    gc.collect()

word2vecpath = PATH + 'embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

with utils.smart_open(word2vecpath) as fin:
    header = utils.to_unicode(fin.readline(), encoding='utf8')
    vocab_size, vector_size = (int(x) for x in header.split())
    binary_len = np.dtype(np.float32).itemsize * vector_size
    for _ in range(vocab_size):
        word = []
        while True:
            ch = fin.read(1)
            if ch == b' ':
                break
            if ch == b'':
                raise EOFError("unexpected end of input")
            if ch != b'\n':
                word.append(ch)
        word = utils.to_unicode(b''.join(word), encoding='utf8', errors='strict')
        weights = np.fromstring(fin.read(binary_len), dtype=np.float32).astype(np.float32)
        word = word.lower()
        if word not in word_index:
            continue
        i = word_index[word]
        if i >= nb_words or embedding_count[i] > 0:
            continue
        global_embedding[i] = weights
        embedding_count[i] += 1

set_seed(seed)

X_test = torch.tensor(X_test, dtype=torch.long).cuda()
test_tensor = torch.utils.data.TensorDataset(X_test)
test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

# TO SAVE FINAL PREDICTIONS
oof_preds = np.zeros(len(X_train))
final_preds = np.zeros(len(X_test))

splits = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed).split(X_train, train_labels)

for fold_number, (train_idx, valid_idx) in enumerate(splits):
    print(f'\nFold {fold_number + 1}\n')

    train_x = torch.tensor(X_train[train_idx], dtype=torch.long).cuda()
    train_y = torch.tensor(train_labels[train_idx, np.newaxis], dtype=torch.float32).cuda()

    valid_x = torch.tensor(X_train[valid_idx], dtype=torch.long).cuda()
    valid_y = torch.tensor(train_labels[valid_idx, np.newaxis], dtype=torch.float32).cuda()

    embedding_matrix = global_embedding.copy()

    model = GRUNet(embedding_matrix)
    model.cuda()

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters())
    
    train_tensor = torch.utils.data.TensorDataset(train_x, train_y)
    valid_tensor = torch.utils.data.TensorDataset(valid_x, valid_y)
    
    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_tensor, batch_size=batch_size, shuffle=False)
    
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)

    test_preds_fold = np.zeros(len(X_test))
    valid_preds_fold = np.zeros((valid_x.size(0)))
    
    for epoch in range(epochs_fixed):
        start_time = time.time()
        model.train()
        avg_loss = 0.
        for x_batch, y_batch in tqdm(train_loader, disable=True):
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        
        model.eval()
        avg_val_loss = 0.
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch).detach()
            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
        
        elapsed_time = time.time() - start_time 
        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
            epoch + 1, epochs_fixed, avg_loss, avg_val_loss, elapsed_time))
        
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    model.embedding.weight.requires_grad = True

    for epoch in range(epochs_trainable):
        start_time = time.time()
        
        model.train()
        avg_loss = 0.
        for x_batch, y_batch in tqdm(train_loader, disable=True):
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        
        model.eval()
        avg_val_loss = 0.
        for x_batch, y_batch in valid_loader:
            y_pred = model(x_batch).detach()
            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
        
        elapsed_time = time.time() - start_time 
        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
            epoch + 1, epochs_trainable, avg_loss, avg_val_loss, elapsed_time))
        
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('checkpoint.pt'))

    for i, (x_batch,) in enumerate(test_loader):
        y_pred = model(x_batch).detach()

        test_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

    model.eval()
    avg_val_loss = 0.
    for i, (x_batch, y_batch) in enumerate(valid_loader):
        y_pred = model(x_batch).detach()
        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
        valid_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

    print('Making oof preds with model with valid loss: {}\n'.format(avg_val_loss))
    oof_preds[valid_idx] = valid_preds_fold
    final_preds += test_preds_fold / num_folds

    del train_x, train_y, valid_x, valid_y, model, embedding_matrix, early_stopping
    gc.collect()

optimal_threshold = threshold_search(train_labels, oof_preds)
print(optimal_threshold)

with open('results_log.txt', 'a') as f:
    f.write(change_string)
    f.write(str(time.localtime()))
    f.write('\n')
    f.write(str(optimal_threshold))
    f.write('\n\n')

# SUBMISSION FILE
final_preds = (final_preds > optimal_threshold['threshold']).astype(int)
sample = pd.read_csv(PATH+'sample_submission.csv')
sample['prediction'] = final_preds
sample.to_csv('submission.csv', index=False)