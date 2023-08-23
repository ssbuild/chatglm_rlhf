# coding=utf8
# @Time    : 2023/5/7 17:28
# @Author  : tk
# @FileName: rlhf_config
import json
import os

import torch
from transformers import BitsAndBytesConfig
from config.constant_map import train_info_models

# 量化权重不支持此模式训练
train_model_config = train_info_models['chatglm']

global_args = {


    # load_in_4bit 量化配置
    "quantization_config": None,
    "num_layers_freeze": -1, # 非lora,非p-tuning 模式 ， <= config.json num_layers
    "pre_seq_len": None,    #p-tuning-v2 参数 , None 禁用p-tuning-v2
    "prefix_projection": False, #p-tuning-v2 参数
    "num_layers": -1, # 是否使用骨干网络的全部层数 最大1-28， -1 表示全层, 否则只用只用N层
}



ppp_info_args = {
    "model_arch_type": "prefixlm" , # one of one of causal, prefixlm,seq2seq
    "ppo_epochs": 2, # Number of updates per batch
    "num_rollouts": 128, # Number  of experiences to observe before learning
    "chunk_size": 1, # Number of chunk_size of training
    "minibatch_size": None,
    "init_kl_coef": 0.001, # Initial value for KL coefficient
    "target": None, # Target value for KL coefficient
    "horizon": 10000, # Number of steps for KL coefficient to reach target
    "gamma": 1., # Discount factor"
    "lam": 0.95, # GAE lambda
    "cliprange": 0.2, # "Clipping range for PPO policy loss (1 - cliprange, 1 + cliprange)"})
                        # cliprange_value: float = field(default=0.2, metadata={"help": "Clipping range for predicted values"
    "cliprange_value": 0.2, # Clipping range for predicted values"
                          #   "(observed values - cliprange_value, observed values + cliprange_value)"}
    "vf_coef": 1., # Value loss scale w.r.t policy loss
    "scale_reward": "ignored",
    "ref_mean": None,
    "ref_std": None,
    "cliprange_reward": 10,
    # Additioanl kwargs for the generation
    "gen_kwargs": dict(
        max_new_tokens=128,
        top_k=0,
        top_p=1.0,
        do_sample=True,
    ),
    "gen_experience_kwargs": None, # Additioanl kwargs for the gen_experience_kwargs

}


train_info_args = {
    'devices': 1,
    'data_backend': 'record',
    'model_type': 'chatglm',
    # 预训练模型路径
    **train_model_config,


    'convert_onnx': False, # 转换onnx模型
    'do_train': True,
    'train_file':  [ './data/train.json'],
    'max_epochs': 20,
    'max_steps': -1,
    'optimizer': 'lion', # one of [lamb,adamw_hf,adamw,adamw_torch,adamw_torch_fused,adamw_torch_xla,adamw_apex_fused,adafactor,adamw_anyprecision,sgd,adagrad,adamw_bnb_8bit,adamw_8bit,lion_8bit,lion_32bit,paged_adamw_32bit,paged_adamw_8bit,paged_lion_32bit,paged_lion_8bit]

    'scheduler_type': 'CAWR', #one of [linear,WarmupCosine,CAWR,CAL,Step,ReduceLROnPlateau, cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau]
    'scheduler':{'T_mult': 1,
             'rewarm_epoch_num': 0.5,  # 如果 max_epochs is not None !
             # 'T_0': 50000,    # 如果 max_epochs is None , 设定步数
             'verbose': False},

    # 'scheduler_type': 'linear',# one of [linear,WarmupCosine,CAWR,CAL,Step,ReduceLROnPlateau
    # 'scheduler': None,

    # 'scheduler_type': 'linear',# one of [linear,WarmupCosine,CAWR,CAL,Step,ReduceLROnPlateau
    # 'scheduler': None,

    # 切换scheduler类型
    # 'scheduler_type': 'WarmupCosine',
    # 'scheduler': None,

    # 'scheduler_type': 'ReduceLROnPlateau',
    # 'scheduler': None,

    # 'scheduler_type': 'Step',
    # 'scheduler':{ 'decay_rate': 0.999,'decay_steps': 100,'verbose': True},

    # 'scheduler_type': 'CAWR',
    # 'scheduler':{'T_mult': 1, 'rewarm_epoch_num': 2, 'verbose': True},

    # 'scheduler_type': 'CAL',
    # 'scheduler': {'rewarm_epoch_num': 2,'verbose': True},

    'optimizer_betas': (0.9, 0.999),
    'train_batch_size': 1,
    'eval_batch_size': 2,
    'test_batch_size': 2,
    'learning_rate': 2e-5,  #
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'max_seq_length':  512, #
    'max_target_length': 100,  # 预测最大长度
    'use_fast_tokenizer': False,
    

    ##############  lora模块
    "ppo": {**ppp_info_args},
}


