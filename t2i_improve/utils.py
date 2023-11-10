# helper functions for Eval&Refine
import re
from compel import Compel
import difflib
import numpy as np
from PIL import Image

def display_alongside_batch(img_list, resize_dims=(256,256)):
    res = np.concatenate([np.array(img.resize(resize_dims)) for img in img_list], axis=1)
    return Image.fromarray(res)

def remove_brackets(text):
    return re.sub(r'\[|\]', '', text)

def get_prompt_decompositions(prompt):
    last_bracket_index = prompt.rfind("]")
    if last_bracket_index != -1:
        prompt = prompt[:last_bracket_index+1]
    out = [phrase.strip("[] ") for phrase in prompt.split("]") if phrase]
    return out

def get_weighted_assertion(assertion, weight):
    if weight==1:
        return assertion
    else:
        out = "({}){}".format(assertion,'+'*weight)
    return out

def get_weighted_prompt(prompt_decompositions,weights):
    weighted_decompositions = [get_weighted_assertion(x,y) for x,y in zip(prompt_decompositions,weights)]
    weighted_prompt = ' '.join(weighted_decompositions)
    return weighted_prompt

def get_prompt_embeddings(prompt_decompositions,weights,compel_proc):
    assert len(prompt_decompositions)==weights.shape[1], "prompt decompositions and weights should be of same length"
    weighted_prompts = [get_weighted_prompt(prompt_decompositions,weights[i]) for i in range(weights.shape[0])]
    prompt_embeds = compel_proc(weighted_prompts)
    return prompt_embeds

def first_true_elements(arr):
    result = [[False] * len(arr[0]) for _ in range(len(arr))]
    for i, row in enumerate(arr):
        for j, val in enumerate(row):
            if val:
                result[i][j] = True
                break
    return np.array(result)

def update_refinement_weights (weights, vqa_scores, vqa_threshold, attn_delta, use_min_idx=True,use_ranking_order=True):
    update_idxs = vqa_scores < vqa_threshold
    if use_min_idx and not use_ranking_order:
        min_idxs = np.zeros_like(vqa_scores)
        min_idxs[np.arange(vqa_scores.shape[0]), np.argmin(vqa_scores,axis=-1)] = 1
        update_idxs = min_idxs
    elif use_ranking_order:
        update_idxs = first_true_elements(update_idxs)
        
    # update weights
    weights = weights + update_idxs * attn_delta
    return weights

def get_entities(parsed_input):
    entities = [x.split()[-1] for x in parsed_input['entities']]
    return entities

def find_closest_matches(entities, input_dict):
    output_dict = {}
    combined_indices = []
    
    for entity in entities:
        closest_match = difflib.get_close_matches(entity, input_dict.keys(), n=1)
        # print (entity, closest_match, input_dict)
        if closest_match:
            output_dict[entity] = input_dict[closest_match[0]]
            combined_indices.extend(input_dict[closest_match[0]])
        else:
            output_dict[entity] = None
    
    return output_dict, combined_indices

def get_token_indices(prompt,entities, pipe):
    entities_keyword = [x.split()[-1] for x in entities]
    token_dict = pipe.get_token_dict(prompt)
    _, token_indices = find_closest_matches(entities_keyword,token_dict)
    return token_indices

def map_entities_to_subphrases(entities, sub_phrases):
    entity_dict = {}
    index_dict = {}
    for i, entity in enumerate(entities):
        entity = entity.split()[-1]
        for j, sub_phrase in enumerate(sub_phrases):
            if entity in sub_phrase:
                entity_dict[entity] = sub_phrase
                index_dict[i] = j
                break
    return entity_dict, index_dict

def get_ca_weights(pw_weights,index_mapping, entities):
    ca_weights = np.zeros((pw_weights.shape[0],len(entities)))
    for i in index_mapping.keys():
        ca_weights[:,i] = pw_weights[:,index_mapping[i]] - 1
    return ca_weights