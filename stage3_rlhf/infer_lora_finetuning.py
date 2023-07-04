# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer_lora_finetuning
import sys
sys.path.append("..")
import os
import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from transformers import HfArgumentParser,PreTrainedTokenizer
from data_utils import train_info_args, NN_DataHelper
from aigc_zoo.model_zoo.chatglm.ppo_model import MyPPOTransformer,LoraArguments,LoraConfig,PPOArguments,PPOConfig
from aigc_zoo.model_zoo.chatglm.llm_model import ChatGLMTokenizer,ChatGLMConfig
from config.rlhf_config import global_args

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_dict(train_info_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args, None, data_args)
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=ChatGLMTokenizer,
                                                              config_class_name=ChatGLMConfig)
    assert tokenizer.eos_token_id == 130005

    ckpt_dir = './best_ckpt'
    config = ChatGLMConfig.from_pretrained(ckpt_dir)
    lora_args = LoraArguments.from_pretrained(ckpt_dir)

    assert lora_args.inference_mode == True

    new_num_tokens = config.vocab_size
    if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
        config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyPPOTransformer(config=config, model_args=model_args, lora_args=lora_args,
                                torch_dtype=config.torch_dtype,
                                new_num_tokens=new_num_tokens,
                                # load_in_8bit=global_args["load_in_8bit"],
                                # # device_map="auto",
                                # device_map = {"":0} # 第一块卡
                                )
    # 加载lora权重
    pl_model.load_sft_weight(ckpt_dir)

    pl_model.eval().half().cuda()

    enable_merge_weight = False
    if enable_merge_weight:
        # 合并lora 权重 保存
        pl_model.save_sft_weight(os.path.join(ckpt_dir, 'pytorch_model_merge.bin'),merge_lora_weight=True)
    else:
        model = pl_model.get_llm_model()

        text = "哪些食物对糖尿病患者有好处?"
        response, history = model.chat(query=text, tokenizer=tokenizer, max_length=2048,
                                          eos_token_id=config.eos_token_id,
                                          do_sample=True, top_p=0.7, temperature=0.95, )
        print('input', text)
        print('output', response)