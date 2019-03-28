# -*- coding: utf-8 -*-
import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

import json
import numpy as np
import logging
import random
#from bert_client import bert_client
import time


logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt", mode='w')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# 计算单个向量归一化后的余弦相似度 0.84
def cosdis(p1, p2):
    return 0.5 + 0.5 * np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))


def query_topk(query_vec, doc_vecs, topk=1):
    # scores = [np.linalg.norm(query_vec, doc_vec) for doc_vec in doc_vecs]
    scores = [cosdis(query_vec, doc_vec) for doc_vec in doc_vecs]

    topk_idx = np.argsort(scores)[::-1][:1]
    idxs = topk_idx[:topk]
    scores = [scores[i] for i in idxs]
    return zip(idxs, scores)


def eval(train_data, test_data, threshold=0.97, save=False):
    train_lst = list(train_data.keys())  # 所有的topic

    # random.shuffle(train_lst)
    logger.info('test set: %d' % len(test_data))

    if save:
        doc_vecs = bert_client.encode(train_lst)  # 将topic过embedding
        np.save('train_vec.npy', doc_vecs)
    else:
        doc_vecs = np.load('train_vec.npy')


    acc_lst = []
    test_list = list(test_data.keys())
    s = time.time()
    if save:
        test_vecs = bert_client.encode(test_list)
        np.save('test_vec.npy', test_vecs)
    else:
        test_vecs = np.load('test_vec.npy')
    print (time.time() - s)

    assert len(test_list) == len(test_vecs)
    predicts = []
    truths = []

    for query_vec, query in zip(test_vecs, test_list):  # 句子的表示,句子本身
        best_idx, max_score = query_topk(query_vec, doc_vecs)[0]  # 找出最合适的topic
        predicts.append(0 if max_score < threshold else train_data[train_lst[best_idx]])  # 大于阈值则预测topic_idx
        truths.append(test_data[query])  # 正确的topic

        if (max_score < threshold and test_data[query] == 0) \
            or (train_data[train_lst[best_idx]] == test_data[query]):  # 如果未达到阈值且的确为负例或者分类正确
            acc_lst.append(1)
        else:
            acc_lst.append(0)

        acc_lst.append(1 if train_data[train_lst[best_idx]] == test_data[query] else 0)
        logger.info(query+'\t'+train_lst[best_idx]+'\t%d'%acc_lst[-1]+'\t%.5f'%max_score)
    labels = [1 if p==t else 0 for p,t in zip(predicts, truths)]
    acc = 1.0 * sum(labels) / len(labels)
    confuse_mat = [1 if p==t and p != 0 else 0 for p, t in zip(predicts, truths)]  # TP
    pp = 1.0 * sum(confuse_mat) / sum([1 if p > 0 else 0 for p in predicts])
    r = 1.0 * sum(confuse_mat) / sum([1 if p > 0 else 0 for p in truths])
    f = 2* pp * r/(pp+r)

    logger.info('acc %.5f\n' % acc)
    print ('acc is %.5f' % acc)
    print ('p r f : %.5f %.5f %.5f'%(pp, r, f))
    return acc


# /Library/Frameworks/Python.framework/Versions/3.7/bin/bert-serving-start -num_worker=1 -model_dir=/Users/zyzhao/Downloads/chinese_L-12_H-768_A-12/

if __name__ == '__main__':
    # lns = [3, 30, 57,84,111,138,165,192,219,246,273]
    # for test_ln in lns:
    #     train_data = json.load(open('data/train/train%d.json'%test_ln))
    #     test_data = json.load(open('data/test/test%d.json'%test_ln))
    #
    #     acc = eval(train_data, test_data)
    #
    #     logger.info('=====================\n\n')

    raw_dict = json.load(open('dtp_test/ano.json'), encoding='utf8')
    train_data = {}
    test_data = {}
    for it in raw_dict:
        for post in raw_dict[it]['post']:
            test_data[post] = len(train_data) + 1
        train_data[raw_dict[it]['post'][1]] = len(train_data) + 1
    noisy_dict = json.load(open('dtp_test/noisy.json'), encoding='utf8')
    for it in noisy_dict:
        for post in noisy_dict[it]['post']:
            test_data[post] = 0
    print (train_data) # 所有的topic
    print (test_data)  # 所有的sentence,value是其topic对应的数值

    eval(train_data, test_data)