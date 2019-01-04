#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'uniphix'
# print ('****************************to be implemented********************************')
import pickle
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<PAD>": 0, "<UNK>": 1}
        self.word2count = {}
        self.index2word = ["<PAD>", "<UNK>",]
        self.word_size = 2
        self.n_words_for_decoder = self.word_size
        #self.stop_word = stopword

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

lang = Lang('123')
with open('lang', 'wb+') as l:
    pickle.dump(lang, l, 0)
