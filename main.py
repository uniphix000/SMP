#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'uniphix'
# print ('****************************to be implemented********************************')

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import json
import logging
import os
import time
import operator
import numpy as np
import pickle

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(levelname)s: %(message)s')
torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device_cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info('device:{0}'.format(device_cuda))

labels = ['website', 'tvchannel', 'lottery', 'chat', 'match',
          'datetime', 'weather', 'bus', 'novel', 'video', 'riddle',
          'calc', 'telephone', 'health', 'contacts', 'epg', 'app', 'music',
          'cookbook', 'stock', 'map', 'message', 'poetry', 'cinemas', 'news',
          'flight', 'translation', 'train', 'schedule', 'radio', 'email']  # 31
label2idx = {label:i for (label, i) in zip(labels, range(len(labels)))}
idx2label = {i:label for (label, i) in label2idx.items()}

def main(run_type='train', run_place='local'):
    # 载入训练数据
    #
    train_data, dev_data = json.load(open('train.json'), encoding='utf8'), \
                           json.load(open('dev.json'), encoding='utf8')
    train_data_size, dev_data_size = len(train_data), len(dev_data)
    train_data_idx, dev_data_idx = [str(i) for i in range(train_data_size)], \
                                   [str(i) for i in range(dev_data_size)]
    train_pairs, dev_pairs = [], []   # [['今天东莞天气如何', 'weather'],...]
    train_querys, dev_querys = [], []
    train_query, dev_query = tokenize(train_data, run_place, 'train'), tokenize(dev_data, run_place, 'dev')

    #
    for query in train_query:
        train_querys.append(query.strip().split('\t'))
    for query in dev_query:
        dev_querys.append(query.strip().split('\t'))
    lang = Lang('zh-cn')  # 词典

    for (query, idx) in zip(train_querys, train_data_idx):
        train_pairs.append([query, train_data[idx]['label']])  # train_pairs:[[['今天', '东莞', '天气', '如何'], 'weather'], ...
        lang.addSentence(query)
    for (query, idx) in zip(dev_querys, dev_data_idx):
        dev_pairs.append([query, dev_data[idx]['label']])
    logging.info('load data! #training data pairs:{0} dev:{1}'.format(len(train_pairs), len(dev_pairs)))
    logging.info('dict generated! dict size:{0}'.format(lang.word_size))

    # word2idx
    train_pairs_idx, dev_pairs_idx = [], []
    for (query, label) in train_pairs:
        query_idx = [lang.word2index.get(word, 1) for word in query]
        train_pairs_idx.append([query_idx, label2idx[label]])  # [[[2, 3, 4, 5], 6], [[6, 7, 8, 9, 10, 11, 12], 20],...
    for (query, label) in dev_pairs:
        query_idx = [lang.word2index.get(word, 1) for word in query]
        dev_pairs_idx.append([query_idx, label2idx[label]])
    # 初始化网络
    lstm = LSTMNet(300, 300, lang)
    lstm = lstm.cuda() if use_cuda else lstm
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, lstm.parameters()),\
            lr=0.00001, amsgrad=True) #, lr=args.lr, weight_decay=)

    # get batch
    if (run_type == 'train'):
        # train
        best_f_score = 0.0
        for loop in range(30):
            total_loss = 0.0
            lstm.train()
            for i in range(len(train_pairs_idx)):
                # train
                input_tensor = torch.tensor(train_pairs_idx[i][0], dtype=torch.long, device=device_cuda)
                label_tensor = torch.tensor(train_pairs_idx[i][1], dtype=torch.long, device=device_cuda)
                loss = lstm.forward(input_tensor, label_tensor)
                loss.backward()
                optimizer.step()
                total_loss += loss
                print ('\repoch:%d num:%.1f%% loss为%.3f ' %(loop, i/len(train_pairs_idx)*100.0 ,total_loss/(i+1)),end=''),  # TODO cool!
                sys.stdout.flush()

            # eval 改用开发集
            predict_box = []
            lstm.eval()
            for i in range(len(dev_pairs_idx)):
                input_tensor = torch.tensor(dev_pairs_idx[i][0], dtype=torch.long, device=device_cuda)
                predict_label = lstm.forward(input_tensor, )
                predict_box.append(idx2label[predict_label.item()])  # 收纳预测结果
            predict_dict = {}
            for it in dev_data:
                predict_dict[it] = {"query": dev_data[it]['query'], "label": predict_box[int(it)]}
            json.dump(predict_dict, open('predict_tmp.json', 'w'), ensure_ascii=False)
            f_score = GetEvalResult()
            if (f_score > best_f_score):
                print ('new record!')
                best_f_score = f_score
                json.dump(predict_dict, open('predict.json', 'w'), ensure_ascii=False)
                torch.save(lstm.state_dict(), 'model')
                print ('new model saved.')
            print ('best score ever:{0}'.format(best_f_score))

    elif (run_type=='eval'):
        lstm.load_state_dict(torch.load('model'))
        predict_box = []
        lstm.eval()
        for i in range(len(dev_pairs_idx)):
            input_tensor = torch.tensor(dev_pairs_idx[i][0], dtype=torch.long, device=device_cuda)
            predict_label = lstm.forward(input_tensor, )
            predict_box.append(idx2label[predict_label.item()])  # 收纳预测结果
        predict_dict = {}
        for it in dev_data:
            predict_dict[it] = {"query": dev_data[it]['query'], "label": predict_box[int(it)]}
        json.dump(predict_dict, open('predict_tmp.json', 'w'), ensure_ascii=False)
        GetEvalResult()
    else:
        raise Exception('run_type error')












class LSTMNet(nn.Module):
    def __init__(self, embed_size, hidden_size, lang, batch_size=1, num_layers=1, label_size=31, dropout=0.5,):
        super(LSTMNet, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = lang.word_size
        self.batch_size = batch_size
        self.label_size = label_size
        self.dropout = dropout
        self.num_layers = num_layers

        self.embedding = self._init_embedding(lang, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size // 2, num_layers=self.num_layers, batch_first=True,
                                bidirectional=True)#, dropout=self.dropout)
        self.linear = nn.Linear(hidden_size, label_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
        self.attn = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.embed_size),
            nn.Tanh(),
            nn.Linear(self.embed_size, self.embed_size),
            nn.Tanh(),
            nn.Linear(self.embed_size, 1)
        )

    def forward(self, input, labels=None):
        input = self.embedding(input).unsqueeze(0)
        h_t, _ = self.lstm(input)  # (b_s, m_l, h_s)
        h_t_ = torch.sum(h_t, dim=1) / h_t.size()[1]
        # h_t_ = h_t[:,-1,:].squeeze(1)  # 仅用最后的向量
        y_t = self.softmax(self.linear(h_t_))

        # context = torch.rand([1,1,300], device=device_cuda)  # 初始化零向量 0.268
        # context_extend = torch.cat([context] * h_t.size()[1], dim=1)  # (b_s, m_l, h_s)
        # score = self.attn(torch.cat((h_t, context_extend), dim=2))  # (b_s, m_l, 1)
        # attn = self.softmax(score.squeeze(2)).unsqueeze(2)
        # h_t_ = torch.sum(h_t * attn, dim=1)
        # y_t = self.softmax(self.linear(h_t_))

        if self.training:
            loss = self.loss(y_t, labels.unsqueeze(0))
            return loss
        elif self.eval:
            _, predict_label = torch.max(y_t, 1)
            return predict_label


    def _init_embedding(self, lang, embed_size):
        if os.path.exists('pretrained'):
            print ('use pretrained embedding...')
            t = open('pretrained', 'rb')
            embeddings = pickle.load(t)
            t.close()
            return nn.Embedding.from_pretrained(torch.tensor(embeddings, device=device_cuda), freeze=True)
        else:
            if os.path.isfile('sgns_merge'):
                print("Loading pretrained embeddings...")
                start = time.time()

                def get_coefs(word, *arr):
                    return word, np.asarray(arr, dtype=np.float32)

                embeddings_dict = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open('sgns_merge', encoding='utf-8') if
                                       len(o.rstrip().rsplit(' ')) != 2)
                print (len(embeddings_dict))

                # print 'no pretrained: '
                embeddings = np.random.randn(lang.word_size, embed_size).astype(np.float32)
                for word, i in lang.word2index.items():
                    embedding_vector = embeddings_dict.get(word)
                    if embedding_vector is not None:
                        embeddings[i] = embedding_vector
                        print('in: ', word)
                    else:
                        print('out: ', word)
                print("took {:.2f} seconds\n".format(time.time() - start))

                t = open('pretrained', 'wb')
                pickle.dump(embeddings, t)
                t.close()
                return nn.Embedding.from_pretrained(torch.tensor(embeddings, device=device_cuda))
            else:
                print ('no pretrained embedding used...')
                return nn.Embedding(lang.word_size, embed_size)







    #json.dump(predict_dict, open(sys.argv[2], 'w'), ensure_ascii=False)


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<PAD>": 0, "<UNK>": 1}
        self.word2count = {}
        self.index2word = ["<PAD>", "<UNK>",]
        self.word_size = 2
        self.n_words_for_decoder = self.word_size

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.word_size
            self.word2count[word] = 1
            self.index2word.append(word)
            self.word_size += 1
        else:
            self.word2count[word] += 1

    def updateDecoderWords(self):
        # 记录Decoder词表的大小
        self.n_words_for_decoder = self.word_size


def GetEvalResult():
    p = os.popen('python eval.py dev.json predict_tmp.json').read()
    print (p)
    if len(p)<37:
        return 0
    elif len(p)==37:
        return float(p.split()[3])


def tokenize(data_dict, run_place, data_type='train'):
    '''
    这里local和coda可以依靠ltp分词而hpc上只能上传分词文件
    convert json file-->querys file-->toeknized querys file
    :param data:
    :return:
    '''
    if (run_place == 'local') | (run_place=='coda'):
        data = ''
        for i in range(len(data_dict)):  # item in data:
            query = data_dict[str(i)]['query']
            data = data + query + '\n'
        with open('to_be_tokenized.txt', 'w', encoding='utf8' ) as f:
            f.write(data)
            f.close()
        p = os.popen('/home/uniphix/ltp-3.4.0/bin/examples/cws_cmdline --input=\'to_be_tokenized.txt\' \
                    --segmentor-model=\'/home/uniphix/ltp_data_v3.4.0/cws.model\' > tokenized.txt').readlines() \
            if run_place == 'local' else \
            os.popen('/bin/ltp-3.4.0/bin/examples/cws_cmdline --input=\'to_be_tokenized.txt\' \
                            --segmentor-model=\'/bin/ltp_data_v3.4.0/cws.model\' > tokenized.txt').readlines()
        with open('tokenized.txt', 'r', encoding='utf8') as ft:
            query = ft.readlines()  # 已经分过词
        return query
    elif (run_place=='hpc'):
        with open('data/'+data_type + '_tokenized.txt', 'r', encoding='utf8') as ft:
            querys = ft.readlines()  # 已经分过词
        return querys




if __name__ == '__main__':
    run_type, run_place = sys.argv[1], sys.argv[2]
    main(run_type, run_place)