# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 9:10

from datasets import load_dataset
import json



def make_json_data():

    ds = load_dataset("openai/summarize_from_feedback", name="comparisons")

    # ds['train'].to_json('./train.json')
    # ds['validation'].to_json('./eval.json')

    with open('./train.json', mode='w', encoding='utf-8', newline='\n') as f:
        for d in ds['train']:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    with open('./eval.json', mode='w', encoding='utf-8', newline='\n') as f:
        for d in ds['validation']:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')


def make_json_data2():
    ds = load_dataset("Dahoas/rm-static")

    with open('./train.json', mode='w', encoding='utf-8', newline='\n') as f:
        for d in ds['train']:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    with open('./eval.json', mode='w', encoding='utf-8', newline='\n') as f:
        for d in ds['test']:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

def make_json_data3():
    ds = load_dataset("zwh9029/rm-static-m2m100-zh-jianti")

    with open('./train.json', mode='w', encoding='utf-8', newline='\n') as f:
        for d in ds['train']:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    with open('./eval.json', mode='w', encoding='utf-8', newline='\n') as f:
        for d in ds['test']:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    make_json_data3()