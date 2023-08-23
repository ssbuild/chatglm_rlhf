# -*- coding: utf-8 -*-
# @Time    : 2023/5/17 11:36
import sys
sys.path.append("..")

import torch
import numpy as np
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from transformers import HfArgumentParser,PreTrainedTokenizer

from config.reward_config import get_deepspeed_config
from data_utils import train_info_args, NN_DataHelper
from models import MyRewardTransformer ,PetlArguments,ChatGLMConfig,ChatGLMTokenizer


deep_config = get_deepspeed_config()

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_dict(train_info_args,allow_extra_keys=True)

    tokenizer : PreTrainedTokenizer
    dataHelper = NN_DataHelper(model_args, None, data_args)
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=ChatGLMTokenizer,
                                                              config_class_name=ChatGLMConfig)

    ckpt_dir = './best_ckpt'
    config = ChatGLMConfig.from_pretrained(ckpt_dir)


    pl_model = MyRewardTransformer(config=config, model_args=model_args)

    if deep_config is None:
        train_weight = './best_ckpt/last-v3.ckpt'
    else:
        # 建议直接使用转换脚本命令 支持 deepspeed stage 0,1,2,3， 生成 ./best_ckpt/last.ckpt/best.pt 权重文件
        # cd best_ckpt/last.ckpt
        # python zero_to_fp32.py . best.pt
        train_weight = './best_ckpt/last.ckpt/best.pt'
        # deepspeed stage 0,1,2 不必须执行上面命令
        # train_weight = './best_ckpt/last.ckpt/checkpoint/mp_rank_00_model_states.pt'

    pl_model.load_sft_weight(train_weight)

    # 保存hf权重
    # config.save_pretrained('convert/')

    # 保存sft p-tuning-v2 权重
    #  pl_model.save_sft_weight('convert/pytorch_model_sft_ptv2.bin')

    # 保存sft权重
    # pl_model.save_sft_weight('convert/pytorch_model_sft.bin')


    pl_model.eval().half().cuda()


    pl_model.requires_grad_(False)

    input_list = [
        "\n\nHuman:如何培养土豆\n\nAssistant:土豆生长在地下,然后发送的干子称为花生,这些花生成长为我们熟悉的土豆。",
        "\n\nHuman:如何培养土豆\n\nAssistant:土豆在地下生长成大、坚固的花生,一旦土豆长大了,它们就生长在地上。",
        "\n\nHuman:火柴是怎样制造的?\n\nAssistant:我猜你问我如何制造某些东西,但我们以前从未真正讨论过制造的细节。",
        "\n\nHuman:火柴是怎样制造的?\n\nAssistant:对不起,我担心我不明白你的问题。",
    ]
    tokend = tokenizer(input_list,padding=True,truncation=True)
    input_ids = torch.tensor(tokend["input_ids"],dtype=torch.int32).to(pl_model.device)
    output = pl_model.backbone.compute_loss(input_ids=input_ids)
    _,scores = output

    for text,score in zip(input_list,scores):
        print('score:' ,score, "text ",text.replace('\n',''))