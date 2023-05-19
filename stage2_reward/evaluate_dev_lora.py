# -*- coding: utf-8 -*-
# @Time    : 2023/5/17 14:26
import json
import os
import numpy as np
import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from tqdm import tqdm
from transformers import HfArgumentParser,PreTrainedTokenizer

from data_utils import train_info_args, NN_DataHelper
from models import MyRewardTransformer, load_in_8bit,LoraArguments,ChatGLMConfig

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, _ = parser.parse_dict(train_info_args)

    tokenizer : PreTrainedTokenizer
    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config()

    ckpt_dir = './best_ckpt'
    config = ChatGLMConfig.from_pretrained(ckpt_dir)
    lora_args = LoraArguments.from_pretrained(ckpt_dir)

    assert lora_args.inference_mode == True

    pl_model = MyRewardTransformer(config=config, model_args=model_args, training_args=training_args,lora_args=lora_args,
                                load_in_8bit=load_in_8bit, device_map="auto")
    # 加载lora权重
    pl_model.backbone.from_pretrained(pl_model.backbone.model, pretrained_model_name_or_path=ckpt_dir, lora_config = lora_args)
    if load_in_8bit:
        pl_model.eval().cuda()
    else:
        pl_model.eval().half().cuda()

    pl_model.requires_grad_(False)

    with open('./data/eval.json',mode='r',encoding='utf-8') as f:
        lines = f.readlines()
    print('predict........')
    acc = 0
    total = 0
    for line in tqdm(lines,total=len(lines)):
        jd = json.loads(line)
        if not jd:
            continue
        input_list = [
            jd['prompt'][:256] + jd['chosen'][:256],
            jd['prompt'][:256] + jd['rejected'][:256],
        ]
        tokend = tokenizer(input_list,padding=True,truncation=True)
        input_ids = torch.tensor(tokend["input_ids"],dtype=torch.int32).to(pl_model.device)
        output = pl_model.backbone.compute_loss(input_ids=input_ids)
        _,scores = output
        total += 1
        if scores[0] >= scores[1]:
           acc += 1

    print('total {} , acc count {} , acc {}'.format(total,acc, acc / total))