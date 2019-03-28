#! /usr/bin/python
# -*- coding: utf-8 -*-
# print ('****************************to be implemented********************************')
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import csv
import os
import codecs
import json
import random
import logging
import argparse
import collections
import shutil
import jieba
from tqdm import tqdm, trange
from gpu_manager import GPUManager
#from sklearn import metrics
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        # 读取单个json文档
        # dicts = json.load(open(input_file), encoding='utf8')

        # 读取多行型json文档
        dicts = []
        with codecs.open(input_file, 'r', 'utf-8') as infs:
            for inf in infs:
                inf = inf.strip()
                dicts.append(json.loads(inf))
        return dicts


class MyPro(DataProcessor):
    '''自定义数据读取方法，针对json文件

    Returns:
        examples: 数据集，包含index、中文文本、类别三个部分
    '''

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_test_examples(
            self._read_json(os.path.join(data_dir, "test.json")), 'test')

    def get_interact_examples(self, data_dir, noisy):
        return self._generate_mixture_dict(os.path.join("dtp_test"), noisy)

    def get_labels(self):
        return ['website', 'tvchannel', 'lottery', 'chat', 'match',
                  'datetime', 'weather', 'bus', 'novel', 'video', 'riddle',
                  'calc', 'telephone', 'health', 'contacts', 'epg', 'app', 'music',
                  'cookbook', 'stock', 'map', 'message', 'poetry', 'cinemas', 'news',
                  'flight', 'translation', 'train', 'schedule', 'radio', 'email']

    def _create_examples(self, dicts, set_type):
        examples = []
        for i in range(len(dicts)):  # train dev规模暂定为1000
            guid = "%s-%s" % (set_type, i)
            text_a, text_b = dicts[i]['sentence1'], dicts[i]['sentence2']
            label = int(dicts[i]['gold_label'])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_test_examples(self, dicts, set_type):
        examples = []
        for i in range(len(dicts)):  # test规模为500
            guid = "%s-%s" % (set_type, i)
            text_a, text_b = dicts[i]['sentence1'], dicts[i]['sentence2']
            label = int(dicts[i]['gold_label'])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_interact_examples(self, item, topic_dict, use_stop_words, set_type='interact'):
        stop_words = json.load(open(os.path.join('data', 'stop_words.json')), encoding='utf8')
        (query, label) = item  # query的正确label
        examples = []
        i = 0
        reverse_topic_dict = {value:key for key,value in topic_dict.items()}  # 必须保持次序才能正确对应label
        for i in range(1, len(topic_dict)+1):  # test规模为500
            topic = reverse_topic_dict[i]
            guid = "%s-%s" % (set_type, i)
            i += 1
            text_a, text_b = self._remove_stop_words(query, stop_words, use_stop_words),\
                             self._remove_stop_words(topic, stop_words, use_stop_words)  # TODO 这里可以尝试改变顺序
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))  # 用于打分,没有label
        #assert len(examples) == 93
        return examples, query, reverse_topic_dict

    def _create_finetune_examples(self, topic_dict, use_stop_words, set_type='interact'):
        stop_words = json.load(open(os.path.join('data', 'stop_words.json')), encoding='utf8')
        examples = []
        reverse_topic_dict = {value:key for key,value in topic_dict.items()}  # 必须保持次序才能正确对应label
        for i in range(1, len(topic_dict)+1):  # test规模为500
            topic = reverse_topic_dict[i]
            guid = "%s-%s" % (set_type, i)
            i += 1
            text_a, text_b = self._remove_stop_words(topic, stop_words, use_stop_words),\
                             self._remove_stop_words(topic, stop_words, use_stop_words)  # TODO 这里可以尝试改变顺序
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=1))  # self fine-tune
        return examples, reverse_topic_dict


    def _remove_stop_words(self, sen, stop_words, use):
        if use:
            return ''.join([w for w in jieba.lcut(sen) if w not in stop_words])
        else:
            return sen



    def _generate_mixture_dict(self, path, noisy):
        raw_dict = json.load(open(os.path.join(path, 'ano_300.json')), encoding='utf8')
        train_data = {}
        test_data = {}
        for it in raw_dict:
            for post in raw_dict[it]['post']:
                test_data[post] = len(train_data) + 1
            train_data[raw_dict[it]['post'][1]] = len(train_data) + 1
        len_positive_test_data = len(test_data)
        print (len(test_data))
        test_box = []
        if noisy:  # 如果负采样
            noisy_dict = json.load(open(os.path.join(path, 'noisy_300.json')), encoding='utf8')
            for it in noisy_dict:
                for post in noisy_dict[it]['post']:  # post
                    test_box.append(post)
        random.shuffle(test_box)
        for neg in test_box[:277*5]:
            test_data[neg] = 0
        print (len(test_data))
        logger.info('-------------------len(train_data): %d, len(test_data): %d-------------------' \
                    %(len(train_data), len(test_data)))
        #print (train_data) # 所有的topic {'另讲一个': 1, '不准睡觉': 2, '你跟谁在睡觉': 3,...
        #print (test_data)  # 所有的sentence,value是其topic对应的数值,负例标签为0 {'讲一个别的': 1, '另讲一个': 1, '再讲一个吧': 1,
        topic_dict,  query_dict = train_data, test_data
        return topic_dict,  query_dict



def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, show_exp=False):
    '''Loads a data file into a list of `InputBatch`s.
    Args:
        examples      : [List] 输入样本，包括question, label, index
        label_list    : [List] 所有可能的类别，可以是int、str等，如['book', 'city', ...]
        max_seq_length: [int] 文本最大长度
        tokenizer     : [Method] 分词方法
    Returns:
        features:
            input_ids  : [ListOf] token的id，在chinese模式中就是每个分词的id，对应一个word vector
            input_mask : [ListOfInt] 真实字符对应1，补全字符对应0
            segment_ids: [ListOfInt] 句子标识符，第一句全为0，第二句全为1
            label_id   : [ListOfInt] 将Label_list转化为相应的id表示
    '''
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label] if example.label is not None else None
        if ex_index < 5 and show_exp:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            #logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def val(model, processor, args, label_list, tokenizer, device):
    '''模型验证

    Args:
        model: 模型
	processor: 数据读取方法
	args: 参数表
	label_list: 所有可能类别
	tokenizer: 分词方法
	device

    Returns:
        f1: F1值
    '''
    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, show_exp=False)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            pred = logits.max(1)[1]
            predict = np.hstack((predict, pred.cpu().numpy()))
            gt = np.hstack((gt, label_ids.cpu().numpy()))

        logits = logits.detach().cpu().numpy()
        # print ('--------------分布为：',torch.nn.functional.softmax(torch.tensor(logits, \
        #                                     dtype=torch.float).to(device)))
        label_ids = label_ids.to('cpu').numpy()

    print(len(gt))
    f1 = np.mean(metrics.f1_score(predict, gt, average=None))
    print(f1)

    return f1


def test(model, processor, args, label_list, tokenizer, device):
    '''模型测试

    Args:
        model: 模型
	processor: 数据读取方法
	args: 参数表
	label_list: 所有可能类别
	tokenizer: 分词方法
	device

    Returns:
        f1: F1值
    '''
    #
    predict_box = []
    #
    test_examples = processor.get_test_examples(args.data_dir)
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    model.eval()
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)  # (1, 2)
            pred = logits.max(1)[1]
            # print('--------------分布为:', torch.nn.functional.softmax(torch.tensor(logits, \
            #                                                                      dtype=torch.float).to(device)))
            # print('--------------预测标签为:',pred)
            predict = np.hstack((predict, pred.cpu().numpy()))
            gt = np.hstack((gt, label_ids.cpu().numpy()))
        #predict_box += pred.tolist()
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()

    f1 = np.mean(metrics.f1_score(predict, gt, average=None))
    print('F1 score in text set is {}'.format(f1))

    return


def interact(model, processor, args, label_list, tokenizer, device, fine_tune=True):
    # TODO
    '''
    使用topic进行self fine tune
    Args:
        model: 模型
	processor: 数据读取方法
	args: 参数表
	label_list: 所有可能类别
	tokenizer: 分词方法
	device

    Returns:
        f1: F1值
    '''
    # 修改label_list
    #
    topic_dict, query_dict = processor.get_interact_examples(args.data_dir, args.use_noisy)  # 得到两个词典

    predicts, raw_predicts, truths = [], [], []
    A, B, C = 0, 0, 0  # 分别对应错误分类， 阈值过高和阈值过低
    yes = 0.0

    if args.self_fine_tune:
        logger.info('*************fine-tune!*************')
    ## self fine-tune  ##
        label_list = [0, 1]

        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            device = torch.device("cuda", args.local_rank)
            n_gpu = 1
            torch.distributed.init_process_group(backend='nccl')
            if args.fp16:
                logger.info("16-bits training currently not supported in distributed training")
                args.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
        logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

        # Prepare optimizer
        if args.fp16:
            param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                               for n, param in model.named_parameters()]
        elif args.optimize_on_cpu:
            param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                               for n, param in model.named_parameters()]
        else:
            param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]


        # prepare data
        examples, reverse_topic_dict = processor._create_finetune_examples(topic_dict, args.use_stop_words)
        train_features = convert_examples_to_features(
            examples, label_list, args.max_seq_length, tokenizer, show_exp=False)
        num_train_steps = int(
            len(examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        t_total = num_train_steps
        if args.local_rank != -1:
            t_total = t_total // torch.distributed.get_world_size()
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)
        logger.info("***** Running training *****", len(examples))
        logger.info("len of examples = %d", len(examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        best_score = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                # label_ids = torch.tensor([f if f<31 else 0 for f in label_ids], dtype=torch.long).to(device)
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                # print ('-------------loss:',loss)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
        checkpoint = {
            'state_dict': model.state_dict()
        }
        torch.save(checkpoint, args.finetune_save_pth)
    else:
        logger.info('*************not fine-tune!*************')






    ## interaction ##
    label_list = range(len(topic_dict) + 1)
    for item in query_dict.items():  # 一句query
        truths.append(item[1])  # 获得正确的label
        examples, query, reverse_topic_dict = processor._create_interact_examples(item, topic_dict, args.use_stop_words)  # 得到len(topic)个数个InputExample构成list
        interact_features = convert_examples_to_features(
                examples, label_list, args.max_seq_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in interact_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in interact_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in interact_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in interact_features], dtype=torch.long)
        interact_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        interact_sampler = SequentialSampler(interact_data)
        interact_dataloader = DataLoader(interact_data, sampler=interact_sampler, batch_size=args.eval_batch_size)

        model.eval()
        predict = np.zeros((0,), dtype=np.int32)
        #gt = np.zeros((0,), dtype=np.int32)
        for input_ids, input_mask, segment_ids, label_ids in interact_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)  # (1, 2)
                #pred = logits[1][1]  # 得到匹配分数 TODO
                pred = np.array(torch.nn.functional.softmax(torch.tensor(logits, \
                                             dtype=torch.float).to(device)[0])[1].cpu())  # 得到预测分数
                predict = np.hstack((predict, pred))
                #gt = np.hstack((gt, label_ids.cpu().numpy()))  # gold target/ query label

        p_value= np.max(predict)  # 最大的数的数值
        #print (type(np.where(predict==p_value)))
        p_label = int(list(np.where(predict==p_value)[0])[0])  # 预测的label,如果发生重复，只取序数最小的
        #print (p_value, p_label)
        if args.use_noisy:
            raw_predicts.append(p_label+1)  # 保存初始预测
            predicts.append(p_label+1 if p_value>0.83 else 0)  # 注意这里为了序号对应需要+1
        else:
            predicts.append(p_label + 1)
            raw_predicts.append(p_label + 1)
        if (predicts[-1] != truths[-1]):
            if truths[-1] != 0:
                if raw_predicts[-1] != truths[-1]:  # 错误分类
                    A += 1
                    print('\nerror type A:错误分类 when encourting:{} while the real topic is :{} \
                    置信概率:{:.3f},(g, p)=({},{})\n'.format(query, reverse_topic_dict[truths[-1]],\
                                                                                   p_value, p_label + 1,
                truths[-1]))  # 阈值过高
                else:
                    B += 1
                    print('\nerror type B:阈值过高 when encourting:{} while the real topic is :{} \
                                        置信概率:{:.3f},(g, p)=({},{})\n'.format(query, reverse_topic_dict[truths[-1]], \
                                                                         p_value, p_label + 1, truths[-1]))
            else:  # 误分负例，阈值过低
                C += 1
                print('\nerror type C:误分负例，阈值过低 when encourting:{} while real tag is negative \
                                置信概率:{:.3f},(g, p)=({},{})\n'.format(query, p_value, p_label + 1,
                                                               truths[-1]))  # 误分
        else:
                yes += 1
        #logits = logits.detach().cpu().numpy()
        #label_ids = label_ids.to('cpu').numpy()

        confuse_mat = [1 if p == t and p != 0 else 0 for p, t in zip(predicts, truths)]  # TP
        pp = 1.0 * sum(confuse_mat) / ((sum([1 if p > 0 else 0 for p in predicts]))+1e-10)
        r = 1.0 * sum(confuse_mat) / (sum([1 if p > 0 else 0 for p in truths])+1e-10)
        f = 2 * pp * r / (pp + r + 1e-10)
        acc = yes/len(predicts)
        #f1 = np.mean(metrics.f1_score(predict, gt, average=None))
        print('\rF1 score in text set is {:.3f}; acc is {:.3f}; A,B,C={},{},{} '.format(f,acc,A,B,C),end=''),
        sys.stdout.flush()

    return


def main():
    # ArgumentParser对象保存了所有必要的信息，用以将命令行参数解析为相应的python数据类型
    parser = argparse.ArgumentParser()

    # required parameters
    # 调用add_argument()向ArgumentParser对象添加命令行参数信息，这些信息告诉ArgumentParser对象如何处理命令行参数
    parser.add_argument("--data_dir",
                        default='/users4/xhu/SMP/similarity_data',
                        #default='/home/uniphix/PycharmProjects/SMP/similarity_data',
                        type=str,
                        # required = True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model",
                        default='bert-base-chinese',
                        type=str,
                        # required = True,
                        help="choose [bert-base-chinese] mode.")
    parser.add_argument("--task_name",
                        default='MyPro',
                        type=str,
                        # required = True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='/users4/xhu/SMP/checkpoints/',
                        #default='/home/uniphix/PycharmProjects/SMP/checkpoints/',
                        type=str,
                        # required = True,
                        help="The output directory where the model checkpoints will be written")
    parser.add_argument("--model_save_pth",
                        default='/users4/xhu/SMP/checkpoints/bert_classification.pth',
                        #default='/home/uniphix/PycharmProjects/SMP/checkpoints/bert_classification.pth',
                        type=str,
                        # required = True,
                        help="The output directory where the model checkpoints will be written")
    parser.add_argument("--finetune_save_pth",
                        default='/users4/xhu/SMP/checkpoints_finetune/bert_classification.pth',
                        # default='/home/uniphix/PycharmProjects/SMP/checkpoints/bert_classification.pth',
                        type=str,
                        # required = True,
                        help="The output directory where the model checkpoints will be written")

    # other parameters
    parser.add_argument("--max_seq_length",
                        default=22,
                        type=int,
                        help="字符串最大长度")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="训练模式")
    parser.add_argument("--do_interact",
                        default=True,
                        action='store_true',
                        help="交互模式")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="验证模式")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="英文字符的大小写转换，对于中文来说没啥用")
    parser.add_argument("--train_batch_size",
                        default=128,
                        type=int,
                        help="训练时batch大小")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="验证时batch大小")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="Adam初始学习步长")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=float,
                        help="训练的epochs次数")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for."
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="用不用CUDA")
    parser.add_argument("--local_rank",
                        default=-1,
                        type=int,
                        help="local_rank for distributed training on gpus.")
    parser.add_argument("--seed",
                        default=777,
                        type=int,
                        help="初始化时的随机数种子")
    parser.add_argument("--gradient_accumulation_steps",
                        default=1,
                        type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--optimize_on_cpu",
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU.")
    parser.add_argument("--fp16",
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit.")
    parser.add_argument("--loss_scale",
                        default=128,
                        type=float,
                        help="Loss scaling, positive power of 2 values can improve fp16 convergence.")
    parser.add_argument("--use_pretrained",
                        default=True,
                        action='store_true',
                        help="是否使用预训练模型")
    parser.add_argument("--use_noisy",
                        default=True,
                        action='store_true',
                        help="是否使用负例")
    parser.add_argument("--use_stop_words",
                        default=True,
                        action='store_true',
                        help="是否使用负例")
    parser.add_argument("--self_fine_tune",
                        default=False,
                        action='store_true',
                        help="是否使用self fine tune")  # fixme

    args = parser.parse_args()
    print ('*'*80)
    print (args)
    print('*' * 80)
    # 对模型输入进行处理的processor，git上可能都是针对英文的processor
    processors = {'mypro': MyPro}
    GPUmanager = GPUManager()
    which_gpu = GPUmanager.auto_choice()
    gpu = "cuda:" + str(which_gpu)
    logger.info('GPU%d Seleted!!!!!!!!!!!!!!!!!!!'%which_gpu)
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(gpu if torch.cuda.is_available() and not args.no_cuda else "cpu")
        #device = torch.device("cuda:1" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = 1
        #n_gpu = torch.cuda.device_count()
    else:
        device = torch.device(gpu, args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # 删除模型文件
    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     shutil.rmtree(args.output_dir)
    #     #raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    # os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    #label_list = label_list
    label_list = [0, 1]  # 31

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                          cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank), num_labels=len(label_list))

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                           for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                           for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)

    # train
    global_step = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, show_exp=False)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        best_score = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                #label_ids = torch.tensor([f if f<31 else 0 for f in label_ids], dtype=torch.long).to(device)
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                #print ('-------------loss:',loss)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()

            f1 = val(model, processor, args, label_list, tokenizer, device)

            if f1 > best_score:
                best_score = f1
                print('*f1 score = {}'.format(f1))
                checkpoint = {
                    'state_dict': model.state_dict()
                }
                torch.save(checkpoint, args.model_save_pth)
            else:
                print('f1 score = {}'.format(f1))

    # test
    if args.use_pretrained:
        model.load_state_dict(torch.load(args.model_save_pth)['state_dict'])
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model,cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank), num_labels=2)
        model.to(device)
    if not args.do_interact:
        test(model, processor, args, label_list, tokenizer, device)
    else:
        interact(model, processor, args, label_list, tokenizer, device)  # 用于测试dtp语料随机生成小语料时的F值
    print (args)  # fixme
    # labels = label_list
    # label2idx = {label: i for (label, i) in zip(labels, range(len(labels)))}
    # idx2label = {i: label for (label, i) in label2idx.items()}
    # predict_box = [idx2label[i] for i in predict_box]
    # dicts = json.load(open(os.path.join(args.data_dir, "test.json")), encoding='utf8', \
    #                   object_pairs_hook=collections.OrderedDict)
    # assert len(dicts)==len(predict_box)
    # predict_dict = {}
    # for (it, idx) in zip(dicts, predict_box):
    #     predict_dict[it] = {"query": dicts[it]['query'], "label": predict_box[int(it)]}
    # json.dump(predict_dict, open('predict_tmp.json', 'w'), ensure_ascii=False, \
    #           sort_keys=True)

if __name__ == '__main__':
    main()