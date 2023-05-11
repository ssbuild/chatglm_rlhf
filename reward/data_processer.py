# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 8:32
import json
import numpy as np
from transformers import PreTrainedTokenizer

class CorpusPreprocess_summery:
    # {
    #     "info": {
    #         "id": "t3_2vwp1w",
    #         "post": "I had a car accident on friday, other party involved was speeding and hit me. but because he denies it it seems like I was wrong because he was supposed to go first under normal circumstances. ( give way road markings ) \n\nbut because it was clear when I checked it I drove on, and when I was almost past the intersection he slammed me in the side near the back seat. and caused me to slide across the road for 2-3 meters hit a street light and then bounce back a meter. both doors completely jammed so i had to climb out the window...\n\ncan I somehow get an investigation going about this to see how fast he had to be driving to get this much force in the collision?\nbecause the damage on my car would suggest that he was driving way faster than the legal limit there. ( which is 50 km/h )\n\nalso another reason why i think he was going way faster than admitted is because he could never have reached the intersection from such a distance as where i could not even see him yet\n\n(pictures of the damage:  ) as you can see with the damage, I am lucky to be alive and unharmed right now... 1ft further forward and it could have been my end...\n\nhelp would be appeciated on this :)",
    #         "title": "Anybody with knowledge of the Dutch law around ? car accident questions.",
    #         "subreddit": "legaladvice"
    #     },
    #     "summaries": [
    #         {
    #             "text": " car accident caused me 2-3m damage to my car both doors totally jammed and driving way faster than usual. need info on what to do with this.. thanks :)",
    #             "policy": "sup4_ppo_rm3_kl10",
    #             "note": "Was the accident caused by driving fast."
    #         },
    #         {
    #             "text": " we suspect other party involved of speeding when he hit me but I can't prove it without an investigation into the damage, how can i get such an investigation ? if at all possible.",
    #             "policy": "ref",
    #             "note": "Unclear what happened."
    #         }
    #     ],
    #     "choice": 1
    # }
    @classmethod
    def process(cls,tokenizer,lines):
        D = []
        for i, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            info = jd['info']
            summaries = jd['summaries']
            choice = jd['choice']
            if len(summaries) != 2 or choice not in (0, 1):
                raise ValueError(
                    f"There should be two summaries with a choice that's either 0 or 1. Received {len(summaries)} summaries and choice={choice}."
                )

            original_text_field = "post" if info["post"] is not None else "article"
            text_a = summaries[choice]["text"] + " " + tokenizer.bos_token + " " + info[original_text_field]
            text_b = summaries[0 if choice == 1 else 1]["text"] + " " + tokenizer.bos_token + " " + info[
                original_text_field]
            D.append((text_a, text_b))
        return D


class CorpusPreprocess:
    # {
    #     "prompt": "\n\nHuman: I eat too much fast food. What's a way to start cooking at home more?\n\nAssistant: The easiest way to start cooking more at home is to cook a few meals a week and keep those meals varied. Planning meals and having a list ready will make cooking much easier. Shopping once a week will also keep your meals varied. Try cooking a few recipes you've never made before. Maybe you could try some Italian, Mexican, or Asian dishes. Cooking more at home is also a great way to save money. You can buy ingredients in bulk and cook large portions, which will last you several days.\n\nHuman: Tell me about buying in bulk. Will that make the process easier?\n\nAssistant: Buying in bulk will definitely make cooking and planning meals easier. The key to buying in bulk is to only buy what you know you will use. Keeping a well-stocked pantry with a variety of dried and canned goods will make cooking easier. Buying in bulk will save you money and give you greater variety in your meals.\n\nHuman: How does one buy in bulk?\n\nAssistant:",
    #     "response": " To buy in bulk, start by shopping at warehouse stores or specialty stores. Make a list of the items you usually use in meals and the amounts you use. Make a note of the unit prices for those items. You can then compare unit prices at warehouse stores to see if they're lower. If they are, buy in bulk from warehouse stores. Also, check online for even bigger savings. Another way to buy in bulk is to join a grocery co-op, which is a group of people who buy food in bulk and then split the cost.",
    #     "chosen": " To buy in bulk, start by shopping at warehouse stores or specialty stores. Make a list of the items you usually use in meals and the amounts you use. Make a note of the unit prices for those items. You can then compare unit prices at warehouse stores to see if they're lower. If they are, buy in bulk from warehouse stores. Also, check online for even bigger savings. Another way to buy in bulk is to join a grocery co-op, which is a group of people who buy food in bulk and then split the cost.",
    #     "rejected": " Buying in bulk is easy. There are stores in your area that offer bulk food. Organic, bulk foods are also widely available online. You can find organic dried pasta, nuts, baking items, and spices in bulk. When shopping for groceries, buy the largest package or container possible to get the largest savings. Organic, bulk food is more expensive than traditional packaged food, but much cheaper than organic grocery stores. You can get many non-perishable items for half price or less when buying in bulk."
    # }
    @classmethod
    def process(cls,tokenizer,lines):
        D = []
        for i, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            prompt = jd['prompt']
            # response = jd['response']
            chosen = jd['chosen']
            rejected = jd['rejected']
            text_a = prompt + chosen
            text_b = prompt + rejected
            if text_a == text_b:
                continue
            D.append((text_a, text_b))
        return D

class TokenIds:
    @classmethod
    def process(cls,pair_data,tokenizer: PreTrainedTokenizer,max_seq_length: int):
        text_a, text_b = pair_data
        assert text_a != text_b , ValueError('data must not be same')

        input_ids = tokenizer.encode(text_a, truncation=True, max_length=max_seq_length)
        input_ids2 = tokenizer.encode(text_b, truncation=True, max_length=max_seq_length)

        ctxlen = input_ids.index(tokenizer.bos_token_id)
        ctxlen2 = input_ids2.index(tokenizer.bos_token_id)

        input_ids = np.asarray(input_ids,dtype=np.int32)
        input_ids2 = np.asarray(input_ids2, dtype=np.int32)
        ctxlen = np.asarray(ctxlen, dtype=np.int32)
        ctxlen2 = np.asarray(ctxlen2, dtype=np.int32)

        if len(input_ids) == len(input_ids2):
            if np.all(input_ids == input_ids2):
                return None
        return {
            "input_ids": input_ids,
            "ctxlen": ctxlen,
            "input_ids2": input_ids2,
            "ctxlen2": ctxlen2
        }