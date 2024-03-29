# -*- coding: utf-8 -*-
# @Time:  11:30
# @Author: tk
from deep_training.data_helper import ModelArguments, DataArguments
from transformers import HfArgumentParser
from config import reward_config
from aigc_zoo.model_zoo.chatglm.reward_model import MyRewardTransformer
from aigc_zoo.model_zoo.chatglm.ppo_model import MyPPOTransformer,PetlArguments,LoraConfig,PPOArguments,PPOConfig
from aigc_zoo.model_zoo.chatglm.llm_model import ChatGLMTokenizer,ChatGLMConfig


def load_reward_model(sft_model_dir,sft_weight_path=None) ->MyRewardTransformer:
    '''
        sft_model_dir:      模型配置路径 ， 路径下需存在config.json
        sft_weight_path:    如果是lora 则是lora 权重路径
                            如果是普通 或者 p-tuning-v2 则是权重文件
    '''

    parser = HfArgumentParser((ModelArguments, DataArguments, PetlArguments))
    model_args, data_args, lora_args = parser.parse_dict(reward_config.train_info_args,allow_extra_keys=True)
    lora_args = lora_args.config
    config = ChatGLMConfig.from_pretrained(sft_model_dir)
    # 加载权重
    lora_args = PetlArguments.from_pretrained(sft_model_dir) if lora_args else None
    pl_module = MyRewardTransformer(config=config,model_args=model_args,lora_args=lora_args)

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
    parser = HfArgumentParser((ModelArguments, DataArguments, PetlArguments))
    model_args, data_args, lora_args = parser.parse_dict(ref_train_info_args,allow_extra_keys=True)
    lora_args = lora_args.config
    config = ChatGLMConfig.from_pretrained(sft_model_dir)
    # 加载权重
    lora_args = PetlArguments.from_pretrained(sft_model_dir) if lora_args else None
    pl_module = MyPPOTransformer(config=config,model_args=model_args,lora_args=lora_args)

    # 加载lora sft 或者 sft 或者 p-tuning-v2 权重
    if lora_args and sft_weight_path is None:
        sft_weight_path = sft_model_dir
    pl_module.load_sft_weight(sft_weight_path)

    pl_module.eval()
    pl_module.requires_grad_(False)
    return pl_module