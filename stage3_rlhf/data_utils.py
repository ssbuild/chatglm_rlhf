# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 17:08

import sys
sys.path.append('..')

import copy
import json
import os
import random
import typing
import numpy as np
import torch
from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, DataArguments
from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject, gfile
from transformers import PreTrainedTokenizer, HfArgumentParser
from data_processer import CorpusPreprocess, TokenIds
from models import LoraArguments,LoraConfig,PPOArguments,PPOConfig,ChatGLMTokenizer,ChatGLMConfig
from config.rlhf_config import *



def preprocess(text):
  return text

def postprocess(text):
  return text


class NN_DataHelper(DataHelper):
    index = 1

    def __init__(self, *args, **kwargs):
        super(NN_DataHelper, self).__init__(*args, **kwargs)


    def on_get_labels(self, files: typing.List[str]):
        D = ['score']
        label2id = {label: i for i, label in enumerate(D)}
        id2label = {i: label for i, label in enumerate(D)}
        return label2id, id2label


    def on_data_ready(self):
        self.index = -1

    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        self.index += 1

        tokenizer: PreTrainedTokenizer
        config = self.config
        max_seq_length = self.max_seq_length_dict[mode]

        ppo_args:PPOConfig = self.external_kwargs['ppo_args']
        max_new_tokens = ppo_args.gen_kwargs['max_new_tokens']
        tokenizer = self.tokenizer

        pair_data = data
        d = TokenIds.process(pair_data,tokenizer,max_seq_length,max_new_tokens)
        if self.index < 3:
            print(d)
        return d

    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        D = []
        for file in files:
            with open(file, mode='r', encoding='utf-8', newline='\n') as f:
                lines = f.readlines()
            d = CorpusPreprocess.process(lines)
            D.extend(d)
        return D

    def collate_fn(self, batch):
        merge_keys = ['input_ids']
        batch = copy.copy(batch)
        o = {
            k: []
            for k in batch[0].keys()
        }
        for i, b in enumerate(batch):
            for k in b:
                o[k].append(copy.deepcopy(b[k]))

        o_pad = {
            k: o[k] for k in merge_keys
        }
        tokenizer: PreTrainedTokenizer = self.tokenizer
        o_pad = tokenizer.pad(o_pad,
                              # max_length=self.data_args.train_max_seq_length,
                              return_tensors="pt")
        for k in o_pad:
            o[k] = o_pad[k]
        return o


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments,PPOArguments))
    model_args, training_args, data_args, lora_args,ppo_args = parser.parse_dict(train_info_args)
    lora_args = lora_args.config
    ppo_args = ppo_args.config

    dataHelper = NN_DataHelper(model_args, training_args, data_args,ppo_args=ppo_args)
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=ChatGLMTokenizer,
                                                                   config_class_name=ChatGLMConfig)
    assert tokenizer.eos_token_id == 130005

    # 缓存数据集
    # 检测是否存在 output/dataset_0-train.record ，不存在则制作数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False,shuffle=True,mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file, shuffle=False,mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file, shuffle=False,mode='test')


    # def shuffle_records(record_filenames, outfile, compression_type='GZIP'):
    #     print('shuffle_records record...')
    #     options = RECORD.TFRecordOptions(compression_type=compression_type)
    #     dataset_reader = Loader.RandomDataset(record_filenames, options=options, with_share_memory=True)
    #     data_size = len(dataset_reader)
    #     all_example = []
    #     for i in tqdm(range(data_size), desc='load records'):
    #         serialized = dataset_reader[i]
    #         all_example.append(serialized)
    #     dataset_reader.close()
    #
    #     shuffle_idx = list(range(data_size))
    #     random.shuffle(shuffle_idx)
    #     writer = WriterObject(outfile, options=options)
    #     for i in tqdm(shuffle_idx, desc='shuffle record'):
    #         example = all_example[i]
    #         writer.write(example)
    #     writer.close()
    #
    #
    # # 对每个record 再次打乱
    # for filename in dataHelper.train_files:
    #     shuffle_records(filename, filename)
