import os

import numpy as np
from keras.preprocessing.text import Tokenizer

from config import config
from src.data_preprocessing import preprocess


def GeneralGenerator(Xs, bs):
    curr = 0
    if bs > len(Xs[0]):
        raise ValueError("batch size too large")
    while True:
        if curr + bs > len(Xs[0]):
            return
        yield [_[curr:curr + bs] for _ in Xs], curr/len(Xs[0]), curr//bs
        curr += bs

class reader(object):

    def __init__(self, dataDir):
        print('preprocessing data...')
        trainIn = preprocess(os.path.join(dataDir, 'train.in'))
        testIn = preprocess(os.path.join(dataDir, 'test.in'))
        trainOut = open(os.path.join(dataDir, 'train.out'), 'r')
        testOut = open(os.path.join(dataDir, 'test.out'), 'r')

        print('tokenizing...')
        self.tokenize(trainIn, testIn)
        trainIn = preprocess(os.path.join(dataDir, 'train.in'))
        testIn = preprocess(os.path.join(dataDir, 'test.in'))

        _word_tensor, _char_tensor, _label_tensor = self.genTensor(trainIn, trainOut)
        #pos_ = np.sum(_label_tensor)
        #neg_ = len(_label_tensor) - pos_
        #dif_ = neg_ - pos_
        #count = 0
        #toDelete = []
        #_ = 0
        #while count < dif_:
            #if _label_tensor[_] == 0:
                #toDelete.append(_)
                #count += 1
            #_ += 1
        #_word_tensor = np.delete(_word_tensor, toDelete, axis=0)
        #_char_tensor = np.delete(_char_tensor, toDelete, axis=0)
        #_label_tensor = np.delete(_label_tensor, toDelete, axis=0)

        np.random.seed(config['split_seed'])
        np.random.shuffle(_word_tensor)
        np.random.seed(config['split_seed'])
        np.random.shuffle(_char_tensor)
        np.random.seed(config['split_seed'])
        np.random.shuffle(_label_tensor)

        val_size = int(config['val_ratio'] * len(_word_tensor))

        self.train_word_tensor = _word_tensor[val_size:]
        self.train_char_tensor = _char_tensor[val_size:]
        self.train_label_tensor = _label_tensor[val_size:]

        print('positive ratio of training set: %f' % np.mean(self.train_label_tensor))

        self.val_word_tensor = _word_tensor[:val_size]
        self.val_char_tensor = _char_tensor[:val_size]
        self.val_label_tensor = _label_tensor[:val_size]

        print('positive ratio of validation set: %f' % np.mean(self.val_label_tensor))

        self.test_word_tensor, self.test_char_tensor, self.test_label_tensor = \
        self.genTensor(testIn, testOut)

        print('positive ratio of test set: %f' % np.mean(self.test_label_tensor))

        print('Data are loaded.')
        print('training set size: %d' % len(self.train_char_tensor))
        print('validation set size: %d' % len(self.val_char_tensor))
        print('test set size: %d' % len(self.test_char_tensor))


    def tokenize(self, trainIn, testIn):
        lines = trainIn.readlines() + testIn.readlines()
        wordTokenizer = Tokenizer()
        wordTokenizer.fit_on_texts(lines)
        lines_ = [_.replace(' ', '').replace('\n', '') for _ in lines]
        charTokenizer = Tokenizer(char_level=True, filters='\n ')
        charTokenizer.fit_on_texts(lines_)

        self.wordTokenizer = wordTokenizer
        self.charTokenizer = charTokenizer

        print('found %d word tokens' % len(self.wordTokenizer.word_counts))
        print('found %d char tokens' % len(self.charTokenizer.word_counts))


    def genTensor(self, dataIn, dataOut):
        lines = dataIn.readlines()
        tokens = self.wordTokenizer.texts_to_sequences(lines)
        tokens_ = []
        reserved_token = 4
        for seq in tokens:
            if len(seq) > config['max_seq_len'] - reserved_token:
                seq = seq[:config['max_seq_len'] - reserved_token]
            tmp = []
            for w in seq:
                if w < config['common_word']:
                    if config['reserve_common_word']:
                        tmp.append(config['max_word_num'] - 3)
                elif w >= config['max_word_num'] - reserved_token:
                    if config['reserve_rare_word']:
                        tmp.append(config['max_word_num'] - 4)
                else:
                    tmp.append(w)
            tmp.extend([config['max_word_num'] - 1] * (config['max_seq_len'] - len(tmp) - 1))
            tmp.append(config['max_word_num'] - 2)
            tokens_.append(tmp)
        wordTensor = np.array(tokens_)

        charTensor = []
        beg_token = len(self.charTokenizer.word_counts)
        end_token = len(self.charTokenizer.word_counts) + 1
        for line in lines:
            tokens = self.charTokenizer.texts_to_sequences(line.split())
            if len(tokens) > config['max_seq_len']:
                tokens = tokens[:config['max_seq_len']]
            tokens.extend([[] for _ in range(config['max_seq_len'] - len(tokens))])
            for idx, w in enumerate(tokens):
                if len(w) > config['max_word_len'] - 2:
                    w = w[:config['max_word_len'] - 2]
                w = [beg_token] + w + [end_token]
                w += [len(self.charTokenizer.word_counts) + 2] * (config['max_word_len'] - len(w))
                tokens[idx] = w
            charTensor.append(tokens)
        charTensor = np.array(charTensor)

        config['char_vocabulary_size'] = len(self.charTokenizer.word_counts) + 3

        labelTensor = dataOut.readlines()
        labelTensor = np.array([int(_[0]) for _ in labelTensor])

        return wordTensor, charTensor, labelTensor

    def trainDataGenerator(self, batch_size):
        pos_idx = np.arange(len(self.train_label_tensor))[self.train_label_tensor == 1]
        neg_idx = np.arange(len(self.train_label_tensor))[self.train_label_tensor == 0]
        assert len(neg_idx) > len(pos_idx)
        np.random.shuffle(neg_idx)
        idx = np.concatenate([neg_idx[:len(pos_idx)], pos_idx])
        np.random.shuffle(idx)
        return GeneralGenerator(
            (self.train_char_tensor[idx], self.train_word_tensor[idx], self.train_label_tensor[idx]),
            batch_size
        )

    def valDataGenerator(self, batch_size=-1):
        pos_idx = np.arange(len(self.val_label_tensor))[self.val_label_tensor == 1]
        neg_idx = np.arange(len(self.val_label_tensor))[self.val_label_tensor == 0]
        assert len(neg_idx) > len(pos_idx)
        np.random.shuffle(neg_idx)
        idx = np.concatenate([neg_idx[:len(pos_idx)], pos_idx])
        np.random.shuffle(idx)
        triple = (self.val_char_tensor[idx], self.val_word_tensor[idx], self.val_label_tensor[idx])
        if batch_size == -1:
            return triple
        return GeneralGenerator(
            triple,
            batch_size
        )

    def testDataGenerator(self, batch_size=-1):
        triple = (self.test_char_tensor, self.test_word_tensor, self.test_label_tensor)
        if batch_size == -1:
            return triple
        return GeneralGenerator(
            triple,
            batch_size
        )

if __name__ == '__main__':
    r = reader('/hdd/data/author')
    r.trainDataGenerator()
    x = 1
