# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 17:08

import sys
sys.path.append('..')

import copy
import logging
import math
import torch
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from deep_training.utils.trainer import SimpleModelCheckpointFabric
from lightning.fabric.strategies import DeepSpeedStrategy
from transformers import HfArgumentParser

from data_utils import NN_DataHelper, train_info_args, get_deepspeed_config
from models import MyPPOTransformer, LoraArguments, LoraConfig, PPOArguments, PPOConfig, load_reward_model, \
    load_ref_model,ChatGLMTokenizer,ChatGLMConfig
from deep_training.nlp.rl.ppo.ppo_trainer import PPOTrainer
from config.rlhf_config import global_args

deepspeed_config = get_deepspeed_config()

class MySimpleModelCheckpoint(SimpleModelCheckpointFabric):
    def __init__(self, *args, **kwargs):
        super(MySimpleModelCheckpoint, self).__init__(*args, **kwargs)
        lora_args:LoraConfig= self.external_kwargs['lora_args']
        if deepspeed_config is not None:
            self.weight_file = './best_ckpt/last.ckpt'
            self.last_weight_file = './last_ckpt/last.ckpt'
        elif lora_args is not None:
            self.weight_file = './best_ckpt'
            self.last_weight_file = './last_ckpt'
        else:
            self.weight_file = './best_ckpt/best.pt'
            self.last_weight_file = './last_ckpt/best.pt'

    def on_save_model(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        lora_args : LoraArguments =  self.external_kwargs['lora_args']
        # 保存权重
        if lora_args is None:
            super(MySimpleModelCheckpoint, self).on_save_model(trainer, pl_module)
        else:
            # 保存最新权重
            logging.info('step {} saving model'.format(trainer.global_step))
            # 保存最新权重
            pl_module.backbone.save_pretrained(self.weight_file)


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments, PPOArguments))
    model_args, training_args, data_args, lora_args, ppo_args = parser.parse_dict(train_info_args)
    lora_args = lora_args.config
    ppo_args = ppo_args.config



    checkpoint_callback = MySimpleModelCheckpoint(
        # monitor="loss",
        save_weights_only=True,
        every_n_epochs=1,
        every_n_train_steps=1000 // training_args.gradient_accumulation_steps,
        # 模型参数
        model_args=model_args,
        training_args=training_args,
        lora_args=lora_args, )

    strategy = 'ddp' if torch.cuda.device_count() >= 1 else 'auto'
    if deepspeed_config is not None and len(deepspeed_config):
        strategy = DeepSpeedStrategy(config=deepspeed_config, )

    dataHelper = NN_DataHelper(model_args, training_args, data_args, ppo_args=ppo_args)
    config_kwargs = {"pre_seq_len": global_args["pre_seq_len"],
                     "prefix_projection": global_args["pre_seq_len"]}
    if global_args["num_layers"] > 0:
        config_kwargs["num_layers"] = global_args["num_layers"]
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=ChatGLMTokenizer,
                                                                   config_class_name=ChatGLMConfig,
                                                                   config_kwargs=config_kwargs)
    assert tokenizer.eos_token_id == 130005
    if config.quantization_bit != 0 and not config.pre_seq_len:
        raise AssertionError("quantization only support ptv2 finetuning")

    if config.quantization_bit != 0 and lora_args is not None:
        raise AssertionError("quantization only support ptv2 finetuning")

    if config.pre_seq_len is not None and lora_args is not None:
        raise ValueError('with lora and ptuning v2 cannot open at the same time')

    assert config.quantization_bit == 0,ValueError('量化权重不支持ppo training')

    trainer = PPOTrainer(
        callbacks=[ checkpoint_callback],
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",
        devices=data_args.devices,
        checkpoint_dir=data_args.output_dir,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        #max_grad_norm=training_args.max_grad_norm,
        strategy=strategy,
        # lora int8 precision='32'
        precision='32' if global_args['load_in_8bit'] else '16',# 可以自行尝试  "32": "32-true", "16": "16-mixed", "bf16": "bf16-mixed"
    )


    # 额外参数
    # checkpoint_callback.tokenizer = tokenizer
    # checkpoint_callback.data_args = data_args
    config.save_pretrained('best_ckpt')

    # 缓存数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file, mixed_data=False, shuffle=True, mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file, mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file, mode='test')


    if trainer.global_rank == 0:
        #加载lora
        pl_reward_model = load_reward_model('../stage2_reward/best_ckpt')

        #加载 全参数微调或者p-tuning-v2权重
        #pl_reward_model = load_reward_model('../stage2_reward/best_ckpt','last.pt')

        reward_device = torch.cuda.device_count() - 1
        pl_reward_model = pl_reward_model.half().to(reward_device)
        reward_batch_size = 48
        delta_reward = True

        def get_reward(samples):
            input = tokenizer(
                samples,
                padding=True,
                truncation=True,
                max_length=data_args.max_seq_length,
                return_tensors="pt",
            ).to(reward_device)

            mbs = reward_batch_size
            out = []
            for i in range(math.ceil(len(samples) / mbs)):
                batch_ixs = slice(i * mbs, (i + 1) * mbs)
                input_ids = input.input_ids[batch_ixs]
                rewards = pl_reward_model.forward_returns(**{
                    "input_ids": input_ids
                })
                out.extend(rewards)
            return torch.hstack(out)

        def reward_fn(samples, prompts, org_labels, **kwargs):
            org_labels = [str(l, encoding='utf-8') for l in org_labels]
            samples = [s + tokenizer.eos_token for s in samples]
            rewards = get_reward(samples)
            if not delta_reward:
                return rewards

            original_samples = [p + o + tokenizer.eos_token for p, o in zip(prompts, org_labels)]
            original_rewards = get_reward(original_samples)
            return rewards - original_rewards
    else:
        reward_fn = None

    pl_model = MyPPOTransformer(config=config,model_args=model_args,training_args=training_args,lora_args=lora_args,ppo_args=ppo_args,
                                load_in_8bit=global_args["load_in_8bit"],device_map={"": trainer.fabric.local_rank} if trainer.world_size > 1 else "auto")

    # 恢复权重继续训练
    # pl_model.load_sft_weight('./best_ckpt/best.pt',is_trainable=True)

    # if config.pre_seq_len is not None:
    #     # P-tuning v2
    #     pl_model.get_llm_model().float()
    #     pl_model.get_llm_model().transformer.prefix_encoder.float()
    # else:
    #     # Finetune
    #     pl_model = pl_model.float()
    pl_model = pl_model.float()

    # pl_ref_model = load_ref_model('../reward/best_ckpt')
    pl_ref_model = copy.deepcopy(pl_model)
    pl_ref_model.eval().half()
    pl_ref_model.requires_grad_(False)



    def dataset_loader_filter_fn(dataset):
        print('*' * 30, 'total', len(dataset))
        return dataset

    train_datasets = dataHelper.load_distributed_random_sampler(
        dataHelper.train_files,
        with_load_memory=True,
        collate_fn=dataHelper.collate_fn,
        # batch_size=training_args.train_batch_size,
        batch_size=ppo_args.chunk_size,
        drop_last=True,  # 多卡建议扔掉
        num_processes=trainer.world_size, process_index=trainer.global_rank,
        dataset_loader_filter_fn=dataset_loader_filter_fn,
        num_workers=0,  # Dataloader num_workers
    )

    if train_datasets is not None:
        trainer.fit(pl_model,
                    ref_model=pl_ref_model,
                    train_loader=train_datasets,
                    tokenizer=tokenizer,
                    reward_fn=reward_fn,
                    ppo_config=ppo_args,
                    stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
                    )

