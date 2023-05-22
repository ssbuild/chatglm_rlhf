# -*- coding: utf-8 -*-
# @Time:  11:30
# @Author: tk
import os
import re
from collections import OrderedDict

import torch
from transformers import PretrainedConfig

from models.chatglm_model import *


class SftWeightMinMax:

    def save_pretrained_merge_lora(self,sft_weight_path: str):
        assert os.path.exists(os.path.dirname(sft_weight_path))
        assert self.lora_args is not None and self.lora_args.with_lora
        lora_model : LoraModel = self.backbone
        model: nn.Module = lora_model.merge_and_unload()
        #保存hf权重，可用infer.py推理
        # torch.save(model.model.state_dict(),weight_path_file)
        model.model.save_pretrained(sft_weight_path)
        return model

    def save_pretrained_merge_lora_and_restore(self, sft_weight_path: str):
        assert os.path.exists(os.path.dirname(sft_weight_path))
        assert self.lora_args is not None and self.lora_args.with_lora
        lora_model: LoraModel = self.backbone
        lora_model.merge_adapter()
        # 保存hf权重，可用infer.py推理
        #torch.save(lora_model.model.model.state_dict(), weight_path_file)
        lora_model.model.model.save_pretrained(sft_weight_path)
        lora_model.unmerge_adapter()

    def load_sft_weight(self, sft_weight_path: str, is_trainable=False, strict=False):
        assert os.path.exists(sft_weight_path)
        if self.lora_args is not None and self.lora_args.with_lora:
            # 加载lora权重
            self.backbone.from_pretrained(self.backbone.model, pretrained_model_name_or_path=sft_weight_path,
                                          is_trainable=is_trainable)
        else:
            weight_dict = torch.load(sft_weight_path)
            weights_dict_new = OrderedDict()
            valid_keys = ['module','state_dict']
            for k in valid_keys:
                if k in weight_dict:
                    weight_dict = weight_dict[k]
                    break
            for k, v in weight_dict.items():
                rm_key = '_TransformerLightningModule__backbone'
                if k.startswith(rm_key):
                    base_model_prefix = self.backbone.base_model_prefix
                    k = re.sub(r'{}.{}.'.format(rm_key,base_model_prefix), '', k)
                weights_dict_new[re.sub(r'_forward_module\.', '', k)] = v
            # 加载sft 或者 p-tuning-v2权重
            self.get_llm_model().load_state_dict(weights_dict_new, strict=strict)

    def save_sft_weight(self,sft_weight_path, merge_lora_weight=False):
        if self.lora_args is not None and self.lora_args.with_lora:
            if merge_lora_weight:
                # lora 合并权重 转换 hf权重
                self.save_pretrained_merge_lora(sft_weight_path)
            else:
                #只保存 lora 权重
                self.backbone.save_pretrained(sft_weight_path)
        else:
            config: PretrainedConfig = self.model.config
            if config.pre_seq_len is not None and config.pre_seq_len > 0:
                # 保存sft p-tuning-v2 权重
                torch.save(self.get_llm_model().state_dict(), sft_weight_path)
            else:
                #保存hf权重
                config.save_pretrained(sft_weight_path)
                self.get_llm_model().save_pretrained(sft_weight_path)

class MyRewardTransformer(MyRewardChatGlmLMHeadModel,SftWeightMinMax, with_pl=True):
    def __init__(self, *args, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        super(MyRewardTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        if lora_args is not None and lora_args.with_lora:
            model = LoraModel(self.backbone, lora_args)
            print('*' * 30, 'lora info')
            model.print_trainable_parameters()
            self.set_model(model, copy_attr=False)

    def get_llm_model(self) -> PreTrainedModel:
        if self.lora_args is not None and self.lora_args.with_lora:
            return self.backbone.model.model
        return self.backbone.model


    def forward_returns(self,*args,**kwargs):
        if self.lora_args is not None and self.lora_args.with_lora:
            model = self.backbone.model
        else:
            model = self.backbone
        return model.forward_returns(*args,**kwargs)


class MyPPOTransformer(MyChatglmModelForCausalPrefixLMWithValueHead,PPOModelLoss,SftWeightMinMax, with_pl=True):
    def __init__(self, *args, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        ppo_args: PPOConfig = kwargs.pop('ppo_args', None)
        super(MyPPOTransformer, self).__init__(*args, **kwargs)

        self.lora_args = lora_args
        self.ppo_config = ppo_args
        if lora_args is not None and lora_args.with_lora:
            model = LoraModel(self.backbone, lora_args)
            print('*' * 30, 'lora info')
            model.print_trainable_parameters()
            self.set_model(model, copy_attr=False)


    def get_llm_model(self) -> ChatGLMForConditionalGeneration:
        if self.lora_args is not None and self.lora_args.with_lora:
            return self.backbone.model.model
        return self.backbone.model

    @torch.no_grad()
    def generate(self,*args,**kwargs):
        return self.get_llm_model().generate(*args,**kwargs)

    def configure_optimizers(self):
        p = self.get_named_parameters(self.backbone)
        training_args = self.training_args
        optimizer = AdamW(p, lr=training_args.learning_rate,
                          eps=training_args.adam_epsilon,
                          betas=training_args.optimizer_betas,
                          weight_decay=training_args.weight_decay)

        # optimizer = Lion(p, lr=training_args.learning_rate,
        #                   betas=training_args.optimizer_betas,
        #                   weight_decay=training_args.weight_decay)

        return optimizer


    def training_step(self,*args, **inputs):
        outputs = self.compute_loss(*args, **inputs)
        return outputs

    def validation_step(self, batch):
        outputs = self.compute_loss(**batch)
        return outputs

    def compute_loss(self, *args, **inputs):
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        #     return self.forward_ppo_loss(*args, **inputs)
        return self.forward_ppo_loss(*args, **inputs)

    def forward_logits_values(self,*args,**kwargs):
        return self.model.forward(*args,**kwargs)






def load_reward_model(sft_model_dir,sft_weight_path=None) ->MyRewardTransformer:
    '''
        sft_model_dir:      模型配置路径 ， 路径下需存在config.json
        sft_weight_path:    如果是lora 则是lora 权重路径
                            如果是普通 或者 p-tuning-v2 则是权重文件
    '''

    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments))
    model_args, data_args, lora_args = parser.parse_dict(reward_config.train_info_args,allow_extra_keys=True)
    lora_args = lora_args.config
    config = ChatGLMConfig.from_pretrained(sft_model_dir)
    # 加载权重
    lora_args = LoraArguments.from_pretrained(sft_model_dir) if lora_args else None
    pl_module = MyRewardTransformer(config=config,model_args=model_args,lora_args=lora_args,allow_extra_keys=True)

    # 加载lora sft 或者 sft 或者 p-tuning-v2 权重
    if lora_args and sft_weight_path is None:
        sft_weight_path = sft_model_dir
    pl_module.load_sft_weight(sft_weight_path)

    pl_module.eval()
    pl_module.requires_grad_(False)
    return pl_module




def load_ref_model(ref_train_info_args,sft_model_dir,sft_weight_path=None) ->MyPPOTransformer:
    '''
        sft_model_dir:      模型配置路径 ， 路径下需存在config.json
        sft_weight_path:    如果是lora 则是lora 权重路径
                            如果是普通 或者 p-tuning-v2 则是权重文件
    '''
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments))
    model_args, data_args, lora_args = parser.parse_dict(ref_train_info_args)
    lora_args = lora_args.config
    config = ChatGLMConfig.from_pretrained(sft_model_dir)
    # 加载权重
    lora_args = LoraArguments.from_pretrained(sft_model_dir) if lora_args else None
    pl_module = MyPPOTransformer(config=config,model_args=model_args,lora_args=lora_args)

    # 加载lora sft 或者 sft 或者 p-tuning-v2 权重
    if lora_args and sft_weight_path is None:
        sft_weight_path = sft_model_dir
    pl_module.load_sft_weight(sft_weight_path)

    pl_module.eval()
    pl_module.requires_grad_(False)
    return pl_module