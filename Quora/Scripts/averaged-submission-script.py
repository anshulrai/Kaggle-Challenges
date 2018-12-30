import gc
import re
import os
import pandas as pd
import numpy as np
import random
from unidecode import unidecode
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import string
import re
import math
import operator
import time
from keras.models import Model, Sequential
from keras import layers
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras import backend as K
from keras import initializers, regularizers, constraints, optimizers
from sklearn.model_selection import StratifiedKFold

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)

# GENERAL HYPERPARAMS
num_folds = 5
seed = 42

# HYPERPARAMS FOR TEXT PROCESSING
max_features = 200000
maxlen = 100

# HYPERPARAMS FOR NN
batch_size = 1024
epochs = 2
embed_size = 300

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

mispell_dict = {"aren't" : "are not",
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
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
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
"we'll":" will",
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

class CyclicLR(Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())

def legacy_round(number, points=0):
    p = 10 ** points
    return float(math.floor((number * p) + math.copysign(0.5, number))) / p

def char_count(text, ignore_spaces=True):
        if ignore_spaces:
            text = text.replace(" ", "")
        return len(text)

def lexicon_count(text):
        count = len(text.split())
        return count
    
def syllable_count(text):
        text = text.lower()
        text = "".join(x for x in text if x not in list(string.punctuation))
        if not text:
            return 0
        count = 0
        vowels = 'aeiouy'
        for word in text.split(' '):
            word = word.strip(".:;?!")
            if len(word) < 1:
                continue
            if word[0] in vowels:
                count +=1
            for index in range(1,len(word)):
                if word[index] in vowels and word[index-1] not in vowels:
                    count +=1
            if word.endswith('e'):
                count -= 1
            if word.endswith('le'):
                count+=1
            if count == 0:
                count +=1
        return count

def sentence_count(text):
        ignore_count = 0
        sentences = re.split(r' *[\.\?!][\'"\)\]]*[ |\n](?=[A-Z])', text)
        for sentence in sentences:
            if lexicon_count(sentence) <= 2:
                ignore_count += 1
        return max(1, len(sentences) - ignore_count)
    
def avg_sentence_length(text):
        try:
            asl = float(lexicon_count(text) / sentence_count(text))
            return legacy_round(asl, 1)
        except ZeroDivisionError:
            return 0.0

def avg_syllables_per_word(text):
        syllable = syllable_count(text)
        words = lexicon_count(text)
        try:
            syllables_per_word = float(syllable) / float(words)
            return legacy_round(syllables_per_word, 1)
        except ZeroDivisionError:
            return 0.0

def avg_letter_per_word(text):
        try:
            letters_per_word = float(
                char_count(text) / lexicon_count(text))
            return legacy_round(letters_per_word, 2)
        except ZeroDivisionError:
            return 0.0

def avg_sentence_per_word(text):
        try:
            sentence_per_word = float(
                sentence_count(text) / lexicon_count(text))
            return legacy_round(sentence_per_word, 2)
        except ZeroDivisionError:
            return 0.0
        
def flesch_reading_ease(text):
        sentence_length = avg_sentence_length(text)
        syllables_per_word = avg_syllables_per_word(text)
        flesch = (
            206.835
            - float(1.015 * sentence_length)
            - float(84.6 * syllables_per_word)
        )
        return legacy_round(flesch, 2)

def flesch_kincaid_grade(text):
        sentence_lenth = avg_sentence_length(text)
        syllables_per_word = avg_syllables_per_word(text)
        flesch = (
            float(0.39 * sentence_lenth)
            + float(11.8 * syllables_per_word)
            - 15.59)
        return legacy_round(flesch, 1)

def polysyllabcount(text):
        count = 0
        for word in text.split():
            wrds = syllable_count(word)
            if wrds >= 3:
                count += 1
        return count

def smog_index(text):
        sentences = sentence_count(text)
        if sentences >= 3:
            try:
                poly_syllab = polysyllabcount(text)
                smog = (
                    (1.043 * (30 * (poly_syllab / sentences)) ** .5)
                    + 3.1291)
                return legacy_round(smog, 1)
            except ZeroDivisionError:
                return 0.0
        else:
            return 0.0

def coleman_liau_index(text):
        letters = legacy_round(avg_letter_per_word(text)*100, 2)
        sentences = legacy_round(avg_sentence_per_word(text)*100, 2)
        coleman = float((0.058 * letters) - (0.296 * sentences) - 15.8)
        return legacy_round(coleman, 2)

def automated_readability_index(text):
        chrs = char_count(text)
        words = lexicon_count(text)
        sentences = sentence_count(text)
        try:
            a = float(chrs)/float(words)
            b = float(words) / float(sentences)
            readability = (
                (4.71 * legacy_round(a, 2))
                + (0.5 * legacy_round(b, 2))
                - 21.43)
            return legacy_round(readability, 1)
        except ZeroDivisionError:
            return 0.0

def linsear_write_formula(text):
        easy_word = 0
        difficult_word = 0
        text_list = text.split()[:100]
        for word in text_list:
            if syllable_count(word) < 3:
                easy_word += 1
            else:
                difficult_word += 1
        text = ' '.join(text_list)
        number = float(
            (easy_word * 1 + difficult_word * 3)
            / sentence_count(text))
        if number <= 20:
            number -= 2
        return number / 2

def f1_score(true,pred):
    #considering sigmoid activation, threshold = 0.5
    pred = K.cast(K.greater(pred,0.5), K.floatx())

    groundPositives = K.sum(true) + K.epsilon()
    correctPositives = K.sum(true * pred) + K.epsilon()
    predictedPositives = K.sum(pred) + K.epsilon()

    precision = correctPositives / predictedPositives
    recall = correctPositives / groundPositives

    m = (2 * precision * recall) / (precision + recall)

    return m

def threshold_search(y_true, y_proba):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.001) 
    F = 2/(1/precision + 1/recall)
    best_score = np.max(F)
    best_th = thresholds[np.argmax(F)]
    search_result = {'threshold': best_th, 'f1': best_score}
    return search_result

def clean_text_for_features(x):
    special_character_removal = re.compile(r'[^A-Za-z\.\-\?\!\,\#\@\% ]',re.IGNORECASE)
    x_ascii = unidecode(x)
    x_clean = special_character_removal.sub('',x_ascii)
    return x_clean

def add_features(df, function_list):
    df['question_text'] = df['question_text'].apply(lambda x:str(x))
    for text_function in function_list:
        df[text_function.__name__] = df['question_text'].apply(lambda x: text_function(str(x)))
    df['total_length'] = df['question_text'].apply(len)
    df['capitals'] = df['question_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/(float(row['total_length'])+1),
                                axis=1)
    df['num_words'] = df['question_text'].str.count('\S+')
    df['num_unique_words'] = df['question_text'].apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / (df['num_words']+1)
    del df['num_unique_words'], df['num_words'], df['capitals'], df['total_length']
    gc.collect()
    return df

def dnn_model(features, embedding_weights):
    features_input = layers.Input(shape=(features.shape[1],))
    inp = layers.Input(shape=(maxlen, ))
    x = layers.Embedding(embedding_weights.shape[0], embedding_weights.shape[1], weights=[embedding_weights], trainable=False)(inp)
        
    x = layers.Bidirectional(layers.CuDNNLSTM(64, kernel_initializer='glorot_normal', return_sequences = True))(x)
    x, x_h, x_c = layers.Bidirectional(layers.CuDNNGRU(64, kernel_initializer='glorot_normal', return_sequences=True, return_state = True))(x)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    
    x = layers.concatenate([avg_pool, x_h, max_pool, features_input])
    x = layers.Dense(32, activation="tanh", kernel_initializer='glorot_normal')(x)
    x = layers.Dense(1, activation="sigmoid", kernel_initializer='glorot_normal')(x)
    
    model = Model(inputs=[inp,features_input], outputs=x)
    adam = optimizers.adam(clipvalue=1.0)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=[f1_score])
    return model

if __name__ == "__main__":
    # TEXT PROCESSING FUNCTIONS
    text_function_list = [char_count, lexicon_count, syllable_count, sentence_count, avg_letter_per_word,
                     avg_sentence_length, avg_sentence_per_word, avg_syllables_per_word, flesch_kincaid_grade, flesch_reading_ease,
                     polysyllabcount, smog_index, coleman_liau_index, automated_readability_index, linsear_write_formula]
    
    # LOAD DATA
    print('>>\t LOADING DATA!')
    train_df = pd.read_csv(PATH+'train.csv', usecols=['question_text', 'target'])
    test_df = pd.read_csv(PATH+'test.csv', usecols = ['question_text'])
    print('>>\t LOADING DATA \t DONE!')
    
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
    
    # LOAD DATA AGAIN FOR FUNCTION CREATION
    train_df = pd.read_csv(PATH+'train.csv', usecols=['question_text', 'target'])
    test_df = pd.read_csv(PATH+'test.csv', usecols = ['question_text'])
    
    # CLEAN DATA
    print('>>\t CLEANING DATA!')
    train_df['question_text'] = train_df['question_text'].apply(lambda x: clean_text_for_features(str(x)))
    test_df['question_text'] = test_df['question_text'].apply(lambda x: clean_text_for_features(str(x)))
    print('>>\t CLEANING DATA \t DONE!')
    
    # CREATE TEXT FEATURES
    print('>>\t CREATING TEXT FEATURES!')
    train_df = add_features(train_df, text_function_list)
    test_df = add_features(test_df, text_function_list)
    
    del text_function_list
    gc.collect()
    
    # SAVE AND PROCESS FEATURES TO SEND TO NN
    train_features = train_df.drop(['question_text', 'target'], axis=1)
    test_features = test_df.drop(['question_text'], axis=1)
    
    del train_df, test_df
    
    ss = StandardScaler()
    ss.fit(np.vstack((train_features, test_features)))
    train_features = ss.transform(train_features)
    test_features = ss.transform(test_features)
    
    del ss
    gc.collect()
    print('>>\t CREATING TEXT FEATURES \t DONE!')
    
    # TOKENIZE TEXT
    print('>>\t TOKENIZING TEXT FEATURES!')
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_sentences) + list(test_sentences))
    
    tokenized_train = tokenizer.texts_to_sequences(train_sentences)
    X_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
    
    tokenized_test = tokenizer.texts_to_sequences(test_sentences)
    X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)
    
    del tokenized_test, tokenized_train, train_sentences, test_sentences
    gc.collect()
    print('>>\t TOKENIZING TEXT FEATURES \t DONE!')
    
    # LIST OF ALL EMBEDDINGS USED
    embedding_list = [PATH+'embeddings/paragram_300_sl999/paragram_300_sl999.txt', 
    PATH+'embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec',
    PATH+'embeddings/glove.840B.300d/glove.840B.300d.txt']
    
    # TO SAVE FINAL PREDICTIONS
    final_preds = np.zeros((X_test.shape[0], 1))
    oof_preds = np.zeros((train_features.shape[0], 1))
    
    word_index = tokenizer.word_index
    
    nb_words = min(max_features, len(word_index)+1)
    
    out_embedding_matrix = np.zeros((nb_words, embed_size))
    
    # MEAN AND STD VALUES FOR EMBEDDINGS
    emb_mean_dict = {'paragram_300_sl999':-0.005324783269315958,
                'wiki-news-300d-1M':-0.0033469984773546457,
                'glove.840B.300d':-0.005838498938828707}

    emb_std_dict = {'paragram_300_sl999':0.4934646189212799,
                'wiki-news-300d-1M':0.10985549539327621,
                'glove.840B.300d':0.4878219664096832}
    
    for EMBEDDING_FILE in embedding_list:
        embedding_name = EMBEDDING_FILE.split('/')[3]
        print('>>\t CREATING EMBEDDINGS FOR {}!'.format(embedding_name))
        emb_mean, emb_std = emb_mean_dict[embedding_name], emb_std_dict[embedding_name]
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore'):
            word, vec = o.split(' ', 1)
            if word not in word_index:
                continue
            i = word_index[word]
            if i >= nb_words:
                continue
            embedding_vector = np.asarray(vec.split(' '), dtype='float32')[:embed_size]
            if len(embedding_vector) == embed_size:
                embedding_matrix[i] = embedding_vector
        gc.collect()
        print('>>\t CREATING EMBEDDINGS FOR {} \t DONE!'.format(embedding_name))
        out_embedding_matrix += embedding_matrix
        del embedding_matrix, embedding_vector
    global_embedding = out_embedding_matrix/3
    del out_embedding_matrix
        
    # FOLDS FOR CV
    folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_features, train_labels)):
            print('FOLD NUMBER {}:'.format(n_fold+1))
            train_x, train_feat, train_y = X_train[train_idx], train_features[train_idx], train_labels[train_idx]
            valid_x, valid_feat, valid_y = X_train[valid_idx], train_features[valid_idx], train_labels[valid_idx]
            
            embedding_matrix = global_embedding.copy()
            
            # TRAIN ON FIXED EMBEDDINGS
            print('>>\t TRAINING FOR FIXED!')
            model = dnn_model(train_feat, embedding_matrix)
            model.fit([train_x, train_feat], train_y, batch_size=batch_size, epochs=epochs, shuffle = True)
            print('>>\t TRAINING EMBEDDINGS FOR FIXED DONE!')

            # TRAIN ON TRAINABLE EMBEDDINGS
            model.layers[1].trainable = True
            adam = optimizers.adam(clipvalue=1.0)
            model.compile(loss='binary_crossentropy',
                              optimizer=adam,
                              metrics=[f1_score])

            print('>>\t TRAINING EMBEDDINGS FOR TRAINABLE!')
            model.fit([train_x, train_feat], train_y, batch_size=batch_size, epochs=epochs, shuffle = True)
            print('>>\t TRAINING EMBEDDINGS FOR TRAINABLE DONE!')

            # PREDICT AND SAVE PREDICTIONS
            print('>>\t PREDICTING!')
            valid_preds = model.predict([valid_x,valid_feat], batch_size=batch_size)
            oof_preds[valid_idx] = valid_preds
            final_preds += model.predict([X_test,test_features], batch_size=batch_size)
            print('>>\t PREDICTING FOR DONE!')
        
            del train_x, train_y, valid_x, valid_y, train_feat, valid_feat, valid_preds, model, embedding_matrix
            gc.collect()
    
    optimal_threshold = threshold_search(train_labels, oof_preds)
    print(optimal_threshold, final_preds)
    
    # SUBMISSION FILE
    final_preds = final_preds/num_folds
    print('>>\t CREATING FINAL SUBMISSION FILE!')
    final_preds = (final_preds > optimal_threshold['threshold']).astype(int)
    sample = pd.read_csv(PATH+'sample_submission.csv')
    sample['prediction'] = final_preds
    sample.to_csv('submission.csv', index=False)
    print('>>\t CREATING FINAL SUBMISSION FILE \t DONE!')