# coding=utf8
# @Time    : 2023/5/12 22:22
# @Author  : tk
# @FileName: chatglm_model
from deep_training.nlp.models.chatglm import ChatGLMForConditionalGeneration
from deep_training.nlp.models.transformer_base import TransformerBase

__all__ = [
    'MyChatGLMForConditionalGeneration',
]


class MyChatGLMForConditionalGeneration(ChatGLMForConditionalGeneration):
    def __init__(self, *args,**kwargs):
        super(MyChatGLMForConditionalGeneration, self).__init__(*args,**kwargs)


