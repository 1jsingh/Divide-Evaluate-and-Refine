import numpy as np
import os,sys
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from lavis.models.base_model import tile

import torch
import torch.nn.functional as F

# import OPENAI-API
import openai
import re


def parse_output(output_str):
    '''
    function for parsing LLM output
    '''

    output = {
        'assertions': [],
        'questions': [],
        'entities':[],
    }

    lines = output_str.split('\n')
    current_key = None

    for line in lines:
        line = line.strip().lower()
        if not line:
            continue

        if current_key is None and ('assertions:' not in line and 'decomposable-caption:' not in line and 'questions:' not in line):
            continue

        if 'assertions:' in line:
            current_key = 'assertions'
        elif 'decomposable-caption:' in line:
            current_key = 'decomposable-caption'
        elif 'questions:' in line:
            current_key = 'questions'
        elif 'entities:' in line:
            current_key = 'entities'
        elif current_key:
                point = re.search(r'\d+\.\s*(.+)', line)
                if point:
                    output[current_key].append(point.group(1))

    return output

def generate_prompt_decomposition(prompt):
    with open('t2i_eval/prompt-decomposition-gen-input') as f:
        message = f.read()
    message = message.format(prompt)
    response = openai.ChatCompletion.create(
    model="gpt-4-1106-preview", # upgraded to latest gpt-4-turbo to allow using fewer incontext examples and LLM tokens
    messages=[
        {"role": "system", "content": "You are ChatGPT, a model which breaksdown complex captions into a decomposable format. Answer as concisely as possible."},
        {"role": "user", "content": message}
    ],
        temperature=0.2
    )
    return response["choices"][0]["message"]["content"]

def generate_questions(prompt):
    decomposable_prompt = generate_prompt_decomposition(prompt)
    with open('t2i_eval/disjoint-question-gen-input') as f:
        message = f.read()
    message = message.format(decomposable_prompt)

    response = openai.ChatCompletion.create(
    model="gpt-4-1106-preview", # upgraded to latest gpt-4-turbo to allow using fewer incontext examples and LLM tokens
    messages=[
        {"role": "system", "content": "You are ChatGPT, a model which breaksdown captions of varying complexity into simpler assertions/questions. Answer as concisely as possible."},
        {"role": "user", "content": message}
    ],
        temperature=0.2
    )

    # Parse Output
    output_str = response["choices"][0]["message"]["content"]
    result = parse_output(output_str)

    # add prompt and decomposable prompt
    result['prompt'] = prompt
    result['decomposable_prompt'] = decomposable_prompt.replace('Decomposable Caption: ','')
    return result['questions'], result

def custom_rank_answers(self, samples, answer_list, num_ans_candidates):
        """
        Your custom implementation for _rank_answers

        Generate the first token of answers using decoder and select ${num_ans_candidates}
        most probable ones. Then select answers from answer list, which start with the probable tokens.
        Lastly, use the selected answers as the ground-truth labels for decoding and calculating LM loss.
        Return the answers that minimize the losses as result.

        """
        answer_candidates = self.tokenizer(
            answer_list, padding="longest", return_tensors="pt"
        ).to(self.device)
        answer_candidates.input_ids[:, 0] = self.tokenizer.bos_token_id

        answer_ids = answer_candidates.input_ids
        answer_atts = answer_candidates.attention_mask

        question_output, _ = self.forward_encoder(samples)
        question_states = question_output.last_hidden_state

        tokenized_question = samples["tokenized_text"]
        question_atts = tokenized_question.attention_mask

        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        start_output = self.text_decoder(
            start_ids,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            return_dict=True,
            reduction="none",
        )
        logits = start_output.logits[:, 0, :]  # first token's logit
        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(
            dim=1, index=answer_first_token
        )
        topk_probs, topk_ids = prob_first_token.topk(num_ans_candidates, dim=1)

        # answer input: [num_question*k, answer_len]
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(
            input_ids == self.tokenizer.pad_token_id, -100
        )

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, num_ans_candidates)
        question_atts = tile(question_atts, 0, num_ans_candidates)

        output = self.text_decoder(
            input_ids,
            attention_mask=input_atts,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            labels=targets_ids,
            return_dict=True,
            reduction="none",
        )

        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(num_ques, num_ans_candidates)

        max_topk_ids = log_probs_sum.argmax(dim=1)
        max_ids = topk_ids[max_topk_ids >= 0, max_topk_ids]
        answers = [answer_list[max_id] for max_id in max_ids]
        topk_probs_ = topk_probs.detach().cpu().numpy()    
        probs = [(topk_probs_[i,0],topk_probs_[i,1]) if max_id==0 else (topk_probs_[i,1],topk_probs_[i,0]) for i,max_id in enumerate(max_ids)]
        return answers, probs

class VQAModel:
    '''BLIP-VQA model for computing DA-Scores'''
    def __init__(self, device='cuda'):
        # Load model and preprocessors
        self.blipvqa_model, self.blipvqa_vis_processors, self.blipvqa_txt_processors = load_model_and_preprocess(
            name="blip_vqa", model_type="vqav2", is_eval=True, device=device
        )
        
        ## Override the _rank_answers method with custom implementation
        self.blipvqa_model._rank_answers = custom_rank_answers.__get__(self.blipvqa_model, type(self.blipvqa_model))
        self.device = device

    def get_score(self, image, question):
        image_ = self.blipvqa_vis_processors["eval"](image).unsqueeze(0).to(self.device)
        question_ = self.blipvqa_txt_processors["eval"](question)
        
        with torch.no_grad():
            vqa_pred = self.blipvqa_model.predict_answers(
                samples={"image": image_, "text_input": question_}, 
                inference_method="rank", 
                answer_list=['yes','no'],
                num_ans_candidates=2
            )
        pos_score, neg_score = vqa_pred[1][0][0], vqa_pred[1][0][1]
        return pos_score, neg_score


def compute_dascores(images, questions, vqa_model, use_neg_scores=False, neg_score_coef=1.0):
    '''
    main function for computing DA-score given list of input images and questions
    '''
    vqa_scores = []
    for image in images:
        # initilize yes/no scores
        pos_scores = []
        neg_scores = []
        # iterate through questions and compute vqa scores
        for question in questions:
            pos_score, neg_score = vqa_model.get_score(image, question)
            if not use_neg_scores:
                neg_score = 0
            pos_scores.append(pos_score)
            neg_scores.append(neg_score)
        
        diff_scores = np.array(pos_scores) - neg_score_coef * np.array(neg_scores)
        vqa_scores.append(diff_scores)

        # compute final da-score as average of the individual assertion alignment scores
        da_score = np.mean(vqa_scores,axis=-1)
    return da_score, np.array(vqa_scores)