# -*- coding: utf-8 -*-
# @Time    : 2023/5/17 11:36

import os
import numpy as np
import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
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

    enable_merge_weight = False

    if enable_merge_weight:
        # 合并lora 权重 保存
        pl_model.save_pretrained_merge_lora(os.path.join(ckpt_dir, 'pytorch_model_merge.bin'))
    else:

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