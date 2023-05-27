
## update information
- [deep_training](https://github.com/ssbuild/deep_training)

```text
    05-27 add qlora transformers>=4.30
```

## install
    python >= 3.10
 - pip install -i https://pypi.org/simple -U -r requirements.txt


## update information
- [deep_training](https://github.com/ssbuild/deep_training)



## weight
- [chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)   支持四种微调方式
- [chatglm-6b-int8](https://huggingface.co/THUDM/chatglm-6b-int8)  目前只支持ptv2微调和推理
- [chatglm-6b-int4](https://huggingface.co/THUDM/chatglm-6b-int4)  目前只支持ptv2微调和推理


## data sample

```json
{
	"prompt": "\n\nHuman:你能描述如何清除笔记本计算机屏幕上的指纹和污垢的步骤吗?\n\nAssistant:是的,也许。为了清洁你的屏幕,你首先需要使用微纤维布或软溼布,轻轻地擦拭屏幕表面。接下来,你会想抓住一个软,无毛巾,微纤维清洁布,轻轻地擦回屏幕上去除指纹和污垢。\n\nHuman:我可以将伊索普罗素酒精喷射到布上,并以这种方式清洗它吗?\n\nAssistant:",
	"response": " 是的,你可以这样做,以帮助布从屏幕上收集更多的污垢。 请确保始终使用干净的软布,而不是一块粗糙的、粗糙的或结构化的材料,并确保它是无毛布的。",
	"chosen": " 是的,你可以这样做,以帮助布从屏幕上收集更多的污垢。 请确保始终使用干净的软布,而不是一块粗糙的、粗糙的或结构化的材料,并确保它是无毛布的。",
	"rejected": " 是的,你可以直接将它喷射到布上。"
}
```





## 切换训练模式配置
    修改 config/__init__.py       “from config.sft_config import *”  切换配置文件
    config/sft_config.py            finetuning
    config/sft_config_lora.py       lora finetuning
    config/sft_config_lora_int4.py  lora int4 finetuning
    config/sft_config_lora_int8.py  lora int8 finetuning
    config/sft_config_ptv2.py       lora p-tuning-v2 finetuning



## infer
    # infer_finetuning.py 推理微调模型
    # infer_lora_finetuning.py 推理微调模型
    # infer_ptuning.py 推理p-tuning-v2微调模型
     python infer_finetuning.py



## training
```text
    #制作数据
    python data_utils.py
    注: num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
    dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)
    
    #训练
    python train.py
```
   

### 单机多卡
```text
可见的前两块卡
train_info_args = {
    'devices': 2,
}

# 第一块 和 第三块卡
train_info_args = {
    'devices': [0,2],
}
```

### 多机多卡训练
```text
例子 3个机器 每个机器 4个卡
修改train.py Trainer num_nodes = 3
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=0 python train.py 
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=1 python train.py 
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=2 python train.py 
```

###  int8高效训练

    lora int8     使用官方标准float 16权重文件，修改modes.py
    load_in_8bit = True 
    修改data_utils.py 对应with_lora=True 

### 混合精度

    Trainer.precision = '16-mixed'

### 超大数据集

    修改data_utils.py "data_backend": "lmdb" 


## lora finetuning

    with_lora

## ptv2 finetuning

    with_prompt

## deepspeed
    启动则将data_utils.py  修改 enable_deepspeed 
    lora 模式暂时不支持deepspeed

## 精度训练
    precision = '16' # 半精度训练 "32": "32-true", "16": "16-mixed", "bf16": "bf16-mixed"


## 友情链接

- [pytorch-task-example](https://github.com/ssbuild/pytorch-task-example)
- [tf-task-example](https://github.com/ssbuild/tf-task-example)
- [chatglm_finetuning](https://github.com/ssbuild/chatglm_finetuning)
- [chatyuan_finetuning](https://github.com/ssbuild/chatyuan_finetuning)
- [llm_finetuning](https://github.com/ssbuild/llm_finetuning)
- [rlhf_llm](https://github.com/ssbuild/rlhf_llm)
- [rlhf_chatglm](https://github.com/ssbuild/rlhf_chatglm)
- [rlhf_chatyuan](https://github.com/ssbuild/rlhf_chatyuan)

## 
    纯粹而干净的代码