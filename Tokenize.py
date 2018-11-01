#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'uniphix'
# print ('****************************to be implemented********************************')
import sys
import json
import os

def tokenize(filename, run_place):
    '''
    convert json file-->querys file-->toeknized querys file
    :param data:
    :return:
    '''
    data_dict = _readfile(filename)
    data = ''
    for i in range(len(data_dict)):# item in data:
        query = data_dict[str(i)]['query']
        data = data  + query + '\n'
    with open('to_be_tokenized.txt', 'w', encoding='utf8' ) as f:
        f.write(data)
        f.close()
    if (run_place=='local'):
        p = os.popen('/home/uniphix/ltp-3.4.0/bin/examples/cws_cmdline --input=\'to_be_tokenized.txt\' \
                    --segmentor-model=\'/home/uniphix/ltp_data_v3.4.0/cws.model\' > tokenized.txt').readlines()
    elif (run_place=='coda'):
        p = os.popen('/bin/ltp-3.4.0/bin/examples/cws_cmdline --input=\'to_be_tokenized.txt\' \
                    --segmentor-model=\'/bin/ltp_data_v3.4.0/cws.model\' > tokenized.txt').readlines()
    # elif (run_place=='hpc'):
    #     p = os.popen('~/bin/ltp-3.2d.0/bin/examples/cws_cmdline --input=\'to_be_tokenized.txt\' \
    #                 --segmentor-model=\'/bin/ltp_data_v3.4.0/cws.model\' > tokenized.txt').readlines()
    with open('tokenized.txt', 'r', encoding='utf8') as ft:
        query = ft.readlines()  # 已经分过词
    return query

def _readfile(filename):
    return json.load(open(filename), encoding='utf8')


if __name__ == '__main__':
    '''
        python Tokenize.py train.json tokenized.txt
        针对一个json文件，提取出待分词部分，写入中间文件，交给ltp分词
    '''
    input_filename, run_place = sys.argv[1], sys.argv[2]
    tokenize(input_filename, run_place)










