# @Time    : 2023/4/19 23:02
# @Author  : tk
# @FileName: data_utils
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
from models import PetlArguments,LoraConfig,ChatGLMTokenizer,ChatGLMConfig
from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject, gfile
from transformers import PreTrainedTokenizer, HfArgumentParser
from data_processer import CorpusPreprocess, TokenIds
from config.reward_config import *
from torch.nn import functional as F

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
        tokenizer = self.tokenizer

        pair_data = data
        d = TokenIds.process(pair_data,tokenizer,max_seq_length)
        if self.index < 3:
            print(d)
        return d

    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        tokenizer = self.tokenizer
        D = []
        for file in files:
            with open(file, mode='r', encoding='utf-8', newline='\n') as f:
                lines = f.readlines()
            d = CorpusPreprocess.process(tokenizer,lines)
            D.extend(d)
        return D

    def collate_fn(self, batch):
        o = {k: [] for k in batch[0].keys()}
        for i, b in enumerate(batch):
            for k in b:
                o[k].append(torch.tensor(b[k]))
        max_len = np.max([len(_) for _ in o['input_ids']])
        flag = False
        if 'input_ids2' in o:
            flag = True
            max_len = np.max([max_len] + [len(_) for _ in o['input_ids2']])

        tokenizer: ChatGLMTokenizer = self.tokenizer
        pad_val = tokenizer.pad_token_id
        def get_mask_position_ids(b_input_ids,ctxlens):
            b_position_ids, b_attention_mask = [], []
            for input_ids, context_length in zip(b_input_ids, ctxlens):
                context_length = context_length.squeeze(dim=-1)
                mask_position = context_length - 1
                position_ids = list(range(context_length)) + [mask_position] * (max_len - context_length)
                block_position_ids = [0] * context_length + list(range(1, max_len - context_length + 1))

                attention_mask = torch.ones((1, max_len, max_len))
                attention_mask = torch.tril(attention_mask)
                attention_mask[..., :context_length] = 1
                attention_mask = (attention_mask < 0.5)

                b_position_ids.append(torch.stack((torch.tensor(position_ids), torch.tensor(block_position_ids))))
                b_attention_mask.append(attention_mask)

            b_attention_mask = torch.stack(b_attention_mask, dim=0)
            b_position_ids = torch.stack(b_position_ids, dim=0)
            return b_attention_mask.bool(),b_position_ids.long()

        o['input_ids'] = torch.stack(
            [F.pad(_, (0, max_len - len(_)), mode='constant', value=pad_val) for _ in o['input_ids']])
        ctxlens = o.pop('ctxlen')
        o['attention_mask'],o['position_ids'] = get_mask_position_ids(o['input_ids'],ctxlens)
        if flag:
            o['input_ids2'] = torch.stack(
                [F.pad(_, (0, max_len - len(_)), mode='constant', value=pad_val) for _ in o['input_ids2']])
            ctxlens2 = o.pop('ctxlen2')
            o['attention_mask2'], o['position_ids2'] = get_mask_position_ids(o['input_ids2'], ctxlens2)
        return o

    def make_dataset_all(self):
        data_args = self.data_args

        # schema for arrow parquet
        schema = None
        # 缓存数据集
        if data_args.do_train:
            self.make_dataset_with_args(data_args.train_file, mixed_data=False, shuffle=True, mode='train',
                                        schema=schema)
        if data_args.do_eval:
            self.make_dataset_with_args(data_args.eval_file, mode='eval', schema=schema)
        if data_args.do_test:
            self.make_dataset_with_args(data_args.test_file, mode='test', schema=schema)


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, PetlArguments))
    model_args, training_args, data_args, lora_args = parser.parse_dict(train_info_args)
    lora_args = lora_args.config

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=ChatGLMTokenizer,
                                                                   config_class_name=ChatGLMConfig)
    assert tokenizer.eos_token_id == 130005


    # 缓存数据集
    dataHelper.make_dataset_all()


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
