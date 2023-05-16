# coding=utf8
# @Time    : 2023/5/12 22:22
# @Author  : tk
# @FileName: chatglm_model

import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.models.transformer_base import TransformerBase

from deep_training.nlp.rl.ppo.configuration import PPOArguments, PPOConfig
from deep_training.nlp.rl.ppo.ppo_module import PPOModelLoss
from deep_training.nlp.utils import configure_optimizers
from torch import nn
from deep_training.nlp.models.lora.v2 import LoraModel, LoraArguments,LoraConfig
from transformers import AdamW
from transformers import PreTrainedModel, HfArgumentParser
from transformers.utils import ModelOutput
from config import reward_config
from deep_training.nlp.models.rl.modeling import ChatglmModelForCausalPrefixLMWithValueHead
from deep_training.nlp.models.chatglm import ChatGLMForConditionalGeneration,ChatGLMConfig
from models.tokenization_chatglm import ChatGLMTokenizer
from deep_training.nlp.optimizer.lion import Lion

#如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
load_in_8bit = False


class MyChatGLMForConditionalGeneration(ChatGLMForConditionalGeneration):
    def __init__(self, *args,**kwargs):
        super(MyChatGLMForConditionalGeneration, self).__init__(*args,**kwargs)


class MyTransformerChatGlmLMHeadModel(TransformerBase):
    def __init__(self, *args,**kwargs):
        # 如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
        load_in_8bit = kwargs.get('load_in_8bit', False)
        if not load_in_8bit:
            kwargs.pop("device_map", None)
        super(MyTransformerChatGlmLMHeadModel, self).__init__(*args,**kwargs)
        self.set_model(self.from_pretrained(MyChatGLMForConditionalGeneration, *args, **kwargs))

        if load_in_8bit:
            setattr(self.model, 'model_parallel', True)
            setattr(self.model, 'is_parallelizable', True)
            self.model.enable_input_require_grads()


class MyRewardChatGlmLMHeadModel(MyTransformerChatGlmLMHeadModel):
    def __init__(self,*args,**kwargs):
        super(MyRewardChatGlmLMHeadModel, self).__init__(*args,**kwargs)
        base_model_prefix = self.base_model_prefix[:-1] if self.base_model_prefix.endswith('_') else self.base_model_prefix
        self.transformer_bone = getattr(self.model,base_model_prefix,None)
        assert self.transformer_bone is not None
        self.score = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def forward_reward(self,**batch):
        state = self.transformer_bone(**batch)[0]
        value = self.score(state)
        return value.squeeze(-1).permute(1, 0).contiguous()


    def forward_loss(self,chosen_ids: torch.Tensor, chosen_values: torch.Tensor,
                     rejected_ids: torch.Tensor, rejected_values: torch.Tensor):
        chosen_mean_scores = []
        rejected_mean_scores = []
        loss = 0.
        # pad_id = torch.tensor(self.config.pad_token_id, dtype=chosen_ids.dtype, device=chosen_values.device)
        for i in range(chosen_ids.size(0)):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_value = chosen_values[i]
            rejected_value = rejected_values[i]

            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen_id == self.config.pad_token_id).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen_id.shape[0]
            r_inds = (rejected_id == self.config.pad_token_id).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected_id.shape[0]
            end_ind = max(c_ind, r_ind)

            # Retrieve first index where trajectories diverge
            divergence_ind = (chosen_id != rejected_id).nonzero()[0]
            assert divergence_ind > 0


            # Index into the correct rewards
            c_truncated_reward = chosen_value[divergence_ind:end_ind]
            r_truncated_reward = rejected_value[divergence_ind:end_ind]

            # Append the last rewards to the list of end scores
            chosen_mean_scores.append(c_truncated_reward[-1])
            rejected_mean_scores.append(r_truncated_reward[-1])

            # Compute loss based on truncated rewards (ignore padding)
            loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()

        loss = loss / chosen_ids.size(0)
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return loss,chosen_mean_scores,rejected_mean_scores

    def forward_value(self,input_ids,values):
        bs = values.size(0)
        seq_len = input_ids.shape[1]
        chosen_mean_scores = [
        ]  # we use this name for consistency with the original forwad function
        for i in range(bs):
            input_id = input_ids[i]
            value = values[i]
            c_inds = (input_id == self.config.pad_token_id).nonzero()
            # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
            c_ind = c_inds[0].item() if len(c_inds) > 0 else seq_len
            chosen_mean_scores.append(value[c_ind - 1])
        return values,torch.stack(chosen_mean_scores)

    def forward_returns(self, **inputs):
        input_ids = inputs['input_ids']
        rewards = self.forward_reward(**inputs)
        ends = torch.argmax((input_ids == self.config.eos_token_id).float(), dim=1).view(-1, 1)
        returns = torch.gather(rewards, 1, ends).squeeze(-1)
        return returns


    def compute_loss(self, *args,return_value_only=False,**batch) -> tuple:
        input_a,input_b = {},{}
        for k,v in batch.items():
            i,k = (input_b,k[:-1]) if k.endswith('2') else (input_a,k)
            i[k] = v

        value_a = self.forward_reward(**input_a)
        if len(input_b) > 0:
            value_b = self.forward_reward(**input_b)
            loss,chosen_mean_scores,rejected_mean_scores = self.forward_loss(input_a["input_ids"],value_a,input_b["input_ids"],value_b)
            loss_dict = {
                "loss": loss,
                "chosen_mean_scores": chosen_mean_scores.mean(),
                "rejected_mean_scores": rejected_mean_scores.mean()
            }
            if self.training:
                return (loss_dict,)
            return (loss,value_a,value_b)
        values,chosen_mean_scores = self.forward_value(batch["input_ids"],value_a)
        if return_value_only:
            return (values,)
        return (values,chosen_mean_scores)



class MyChatglmModelForCausalPrefixLMWithValueHead(ChatglmModelForCausalPrefixLMWithValueHead):
    def __init__(self, *args, **kwargs):
        # 如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
        load_in_8bit = kwargs.get('load_in_8bit', False)
        if not load_in_8bit:
            kwargs.pop("device_map", None)
        super(MyChatglmModelForCausalPrefixLMWithValueHead, self).__init__(*args, **kwargs)
        self.set_model(self.from_pretrained(MyChatGLMForConditionalGeneration, *args, **kwargs))

        if load_in_8bit:
            setattr(self.model, 'model_parallel', True)
            setattr(self.model, 'is_parallelizable', True)
            self.model.enable_input_require_grads()