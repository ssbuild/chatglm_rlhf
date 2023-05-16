# -*- coding: utf-8 -*-
# @Time:  11:30
# @Author: tk

from models.chatglm_model import *


class MyRewardTransformer(MyRewardChatGlmLMHeadModel, with_pl=True):
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


class MyPPOTransformer(MyChatglmModelForCausalPrefixLMWithValueHead,PPOModelLoss, with_pl=True):
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
        return self.forward_ppo_loss(*args, **inputs)


    def forward_logits_values(self,*args,**kwargs):
        return self.model.forward(*args,**kwargs)





def load_reward_model(model_dir) ->MyRewardTransformer:
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, lora_args = parser.parse_dict(reward_config.train_info_args)
    lora_args = lora_args.config
    config = ChatGLMConfig.from_pretrained(model_dir)
    if lora_args is not None:
        lora_args = LoraArguments.from_pretrained(model_dir)
    # 加载权重
    pl_module = MyRewardTransformer(config=config,model_args=model_args,training_args=training_args,lora_args=lora_args)
    if lora_args is not None:
        # 加载lora权重
        pl_module.backbone.from_pretrained(pl_module.backbone.model, pretrained_model_name_or_path=model_dir,lora_config=lora_args)
    pl_module.eval()
    pl_module.requires_grad_(False)
    return pl_module


def load_ref_model(lora_model_dir,ref_train_info_args) ->MyPPOTransformer:
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, lora_args = parser.parse_dict(ref_train_info_args)
    lora_args = lora_args.config
    config = ChatGLMConfig.from_pretrained(lora_model_dir)
    if lora_args is not None:
        lora_args = LoraArguments.from_pretrained(lora_model_dir)
    pl_module = MyPPOTransformer(config=config,model_args=model_args,training_args=training_args,lora_args=lora_args)
    if lora_args is not None:
        # 加载lora权重
        pl_module.backbone.from_pretrained(pl_module.backbone.model, pretrained_model_name_or_path=lora_model_dir,lora_config=lora_args)
    pl_module.eval()
    pl_module.requires_grad_(False)
    return pl_module