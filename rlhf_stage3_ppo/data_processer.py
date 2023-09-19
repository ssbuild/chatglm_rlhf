# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 17:09

import json

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer



class CorpusPreprocess:
    # {
    #     "prompt": "\n\nHuman: I eat too much fast food. What's a way to start cooking at home more?\n\nAssistant: The easiest way to start cooking more at home is to cook a few meals a week and keep those meals varied. Planning meals and having a list ready will make cooking much easier. Shopping once a week will also keep your meals varied. Try cooking a few recipes you've never made before. Maybe you could try some Italian, Mexican, or Asian dishes. Cooking more at home is also a great way to save money. You can buy ingredients in bulk and cook large portions, which will last you several days.\n\nHuman: Tell me about buying in bulk. Will that make the process easier?\n\nAssistant: Buying in bulk will definitely make cooking and planning meals easier. The key to buying in bulk is to only buy what you know you will use. Keeping a well-stocked pantry with a variety of dried and canned goods will make cooking easier. Buying in bulk will save you money and give you greater variety in your meals.\n\nHuman: How does one buy in bulk?\n\nAssistant:",
    #     "response": " To buy in bulk, start by shopping at warehouse stores or specialty stores. Make a list of the items you usually use in meals and the amounts you use. Make a note of the unit prices for those items. You can then compare unit prices at warehouse stores to see if they're lower. If they are, buy in bulk from warehouse stores. Also, check online for even bigger savings. Another way to buy in bulk is to join a grocery co-op, which is a group of people who buy food in bulk and then split the cost.",
    #     "chosen": " To buy in bulk, start by shopping at warehouse stores or specialty stores. Make a list of the items you usually use in meals and the amounts you use. Make a note of the unit prices for those items. You can then compare unit prices at warehouse stores to see if they're lower. If they are, buy in bulk from warehouse stores. Also, check online for even bigger savings. Another way to buy in bulk is to join a grocery co-op, which is a group of people who buy food in bulk and then split the cost.",
    #     "rejected": " Buying in bulk is easy. There are stores in your area that offer bulk food. Organic, bulk foods are also widely available online. You can find organic dried pasta, nuts, baking items, and spices in bulk. When shopping for groceries, buy the largest package or container possible to get the largest savings. Organic, bulk food is more expensive than traditional packaged food, but much cheaper than organic grocery stores. You can get many non-perishable items for half price or less when buying in bulk."
    # }
    @classmethod
    def process(cls,lines):
        D = []
        for i, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            prompt = jd['prompt']
            # response = jd['response']
            chosen = jd['chosen']
            rejected = jd['rejected']
            if chosen == rejected:
                print('warning text_a == text_b and it will be ingored')
                continue
            D.append((prompt, chosen))
        return D


class TokenIds:

    # @classmethod
    # def get_prompt(cls,prompt,tokenizer: PreTrainedTokenizer,max_seq_length: int):
    #     """
    #     Get the prompt after T5 decoding to make sure dictionary
    #     of prompts and summaries is consistent decode prompt from trlX pipeline
    #     """
    #     assert max_seq_length > 5
    #     tmp = tokenizer.decode(
    #         tokenizer(
    #             prompt.split("TL;DR:")[0],
    #             truncation=True,
    #             max_length=max_seq_length - 5,  # to make sure "TL;DR" dont get truncated
    #             add_special_tokens=False,
    #         )["input_ids"],
    #         skip_special_tokens=True,
    #     ).strip()
    #     tmp = tmp + "\nTL;DR:"
    #     formatted_prompt = tokenizer.decode(
    #         tokenizer(tmp, truncation=True, max_length=max_seq_length, add_special_tokens=False)["input_ids"],
    #         skip_special_tokens=True,
    #     ).strip()
    #     return formatted_prompt

    @classmethod
    def process(cls,pair_data,tokenizer: PreTrainedTokenizer,max_seq_length: int,max_new_tokens: int):
        prompt, labels = pair_data

        max_prompt_length = max_seq_length - max_new_tokens

        input_ids = tokenizer.encode(prompt, truncation=True,padding=False, max_length=max_prompt_length)
        return {
            "prompt": np.array(bytes(prompt,encoding='utf-8')),
            "org_labels": np.array(bytes(labels, encoding='utf-8')),
            "input_ids": np.asarray(input_ids,dtype=np.int32),
        }