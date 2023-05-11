
## 注意事项
    llama 要求transformers >= 4.28 llama 词典等下载地址 https://huggingface.co/hf-internal-testing/llama-tokenizer
    python >= 3.10

## 1. 安装
    pip install --pre -U deep_training>=0.1.5


## 2.更新详情
- [deep_training](https://github.com/ssbuild/deep_training)

## 3.深度学习常规任务例子

- [pytorch-task-example](https://github.com/ssbuild/pytorch-task-example)
- [tf-task-example](https://github.com/ssbuild/tf-task-example)
- [llm_finetuning](https://github.com/ssbuild/llm_finetuning)
- [llm_rlhf_training](https://github.com/ssbuild/llm_rlhf_training)
- [chatglm_rlhf_training](https://github.com/ssbuild/chatglm_rlhf_training)



## 4. 预训练模型下载，大小尺寸可自行选择
    
- [bloom预训练模型](https://huggingface.co/bigscience)
- [bloom第三方中文训练模型](https://huggingface.co/Langboat/bloom-6b4-zh)  # 注意 需要修改tokenizer_config.json BloomTokenizer -> BloomTokenizerFast
- [opt预训练模型](https://huggingface.co/facebook)
- [llama预训练模型](https://huggingface.co/decapoda-research)


## 数据示例

```json
{
	"prompt": "\n\nHuman:你能描述如何清除笔记本计算机屏幕上的指纹和污垢的步骤吗?\n\nAssistant:是的,也许。为了清洁你的屏幕,你首先需要使用微纤维布或软溼布,轻轻地擦拭屏幕表面。接下来,你会想抓住一个软,无毛巾,微纤维清洁布,轻轻地擦回屏幕上去除指纹和污垢。\n\nHuman:我可以将伊索普罗素酒精喷射到布上,并以这种方式清洗它吗?\n\nAssistant:",
	"response": " 是的,你可以这样做,以帮助布从屏幕上收集更多的污垢。 请确保始终使用干净的软布,而不是一块粗糙的、粗糙的或结构化的材料,并确保它是无毛布的。",
	"chosen": " 是的,你可以这样做,以帮助布从屏幕上收集更多的污垢。 请确保始终使用干净的软布,而不是一块粗糙的、粗糙的或结构化的材料,并确保它是无毛布的。",
	"rejected": " 是的,你可以直接将它喷射到布上。"
}
```