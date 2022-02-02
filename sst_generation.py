#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""


import os
import argparse
import logging

import math
import statistics
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    OpenAIGPTTokenizer,
    OpenAIGPTLMHeadModel,
)

from gpt2_sentiment import SSTModel
from get_distinct_score import eval_distinct


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer, GPT2Config),
}

EOS_ID = 50256
ATTR_TO_SPECIAL_TOKEN = {'pad_token': '<pad>'}
domain_token = "<movie>"
pos_token = "<pos>"
neg_token = "<neg>"
SMALL_CONST = 1e-5

original_tokenizer_len = 50257


def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def generate(model, input_ids, label_ids, cur_len, max_length, do_sample, temperature, top_k, top_p,
             repetition_penalty, batch_size, pad_token_id=None, eos_token_ids=None, use_domain_tag=False, use_label_tag=False,
             domain_ids=None, bayes=False, bayes_gamma=-1, use_mlp=False, bayes_lambda=1):

    unfinished_sents = input_ids.new(batch_size).fill_(1)
    # print("beginning")
    # print(unfinished_sents)
    # print()
    sent_lengths = input_ids.new(batch_size).fill_(max_length)

    with torch.no_grad():
        if (bayes and bayes_gamma > -1) or (use_domain_tag and use_label_tag):
            if use_mlp:
                domain_emb = model.lm_model.transformer.wte(domain_ids)  # bze x emb_size
                label_emb = model.lm_model.transformer.wte(label_ids)  # bze x emb_size
                n_layers = len(model.lm_model.transformer.h)
                domain_emb_list = [domain_emb] * n_layers
                label_emb_list = [label_emb] * n_layers
                label_transfered_past = tuple(model.mlp_transfer_past(label_emb_l, label_mlp_past_transfer_l) for
                                              (label_emb_l, label_mlp_past_transfer_l) in
                                              zip(label_emb_list, model.label_mlp))
                domain_transfered_past = tuple(model.mlp_transfer_past(domain_emb_l, domain_mlp_past_transfer_l) for
                                               (domain_emb_l, domain_mlp_past_transfer_l) in
                                               zip(domain_emb_list, model.domain_mlp))
            else:
                domain_logits, domain_past = model.lm_model(domain_ids, past=None)
                label_logits, label_past = model.lm_model(label_ids, past=None)
                domain_transfered_past = tuple(
                    model.transfer_past(domain_past_l, past_transfer_l) for (domain_past_l, past_transfer_l) in
                    zip(domain_past, model.domain_past_transfer))
                label_transfered_past = tuple(
                    model.transfer_past(label_past_l, past_transfer_l) for (label_past_l, past_transfer_l) in
                    zip(label_past, model.label_past_transfer))

            if bayes and bayes_gamma > -1:
                past = label_transfered_past  # used to use label as the nominator in bayes
            else:
                past = tuple(torch.cat((domain_transfered_past_i, label_transfered_past_i), dim=-2) for
                                    (domain_transfered_past_i, label_transfered_past_i) in
                                    zip(domain_transfered_past, label_transfered_past))

        elif use_domain_tag:
            if use_mlp:
                domain_emb = model.lm_model.transformer.wte(domain_ids)  # bze x emb_size
                n_layers = len(model.lm_model.transformer.h)
                domain_emb_list = [domain_emb] * n_layers
                past = tuple(model.mlp_transfer_past(domain_emb_l, domain_mlp_past_transfer_l) for
                             (domain_emb_l, domain_mlp_past_transfer_l) in
                             zip(domain_emb_list, model.domain_mlp))
            else:
                domain_logits, domain_past = model.lm_model(domain_ids, past=None)
                past = tuple(
                    model.transfer_past(domain_past_l, past_transfer_l) for (domain_past_l, past_transfer_l) in
                    zip(domain_past, model.domain_past_transfer))
        elif use_label_tag:
            if use_mlp:
                label_emb = model.lm_model.transformer.wte(label_ids)  # bze x emb_size
                n_layers = len(model.lm_model.transformer.h)
                label_emb_list = [label_emb] * n_layers
                past = tuple(model.mlp_transfer_past(label_emb_l, label_mlp_past_transfer_l) for
                             (label_emb_l, label_mlp_past_transfer_l) in
                             zip(label_emb_list, model.label_mlp))
            else:
                label_logits, lable_past = model.lm_model(label_ids, past=None)
                past = tuple(
                    model.transfer_past(label_past_l, past_transfer_l) for (label_past_l, past_transfer_l)
                    in zip(lable_past, model.label_past_transfer))
        else:
            past = None

        cur_token = input_ids
        # print(cur_token.shape)
        # print(label_ids)

        past_length = 0
        device = input_ids.device
        
        generated_length = 0

        while cur_len < max_length:
            # print(cur_len)
            # if past is not None:
            #     print(past[0].shape)
            #     print(cur_token.shape)

            # specify position ids (starts with 0)
            input_shape = cur_token.size()
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
            past_length += input_shape[-1]

            logits, past = model.lm_model(cur_token, past=past, position_ids=position_ids)

            next_token_logits = logits[:, -1, :-4]  # the last 4 tokens are special tokens, not considered during generation

            if bayes and (bayes_gamma > -1 and generated_length < bayes_gamma):
                d_logits, domain_transfered_past = model.lm_model(cur_token, past=domain_transfered_past,
                                                                  position_ids=position_ids)
                d_next_token_logits = d_logits[:, -1, :-4]

            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[i, previous_token] *= repetition_penalty
                            if bayes and (bayes_gamma > -1 and generated_length < bayes_gamma):
                                d_next_token_logits[i, previous_token] *= repetition_penalty
                        else:
                            next_token_logits[i, previous_token] /= repetition_penalty
                            if bayes and (bayes_gamma > -1 and generated_length < bayes_gamma):
                                d_next_token_logits[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                    if bayes and (bayes_gamma > -1 and generated_length < bayes_gamma):
                        d_next_token_logits = d_next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p, filter_value=-1e4)

                # Sample
                if bayes and (bayes_gamma > -1 and generated_length < bayes_gamma):
                    d_next_token_logits = top_k_top_p_filtering(d_next_token_logits, top_k=top_k, top_p=top_p, filter_value=-1e4)

                    # # use log space
                    log_next_token_probs = F.log_softmax(next_token_logits, dim=-1)
                    log_d_next_token_probs = F.log_softmax(d_next_token_logits, dim=-1)
                    log_d_next_token_probs[log_d_next_token_probs < -5] = 0
                    log_bayes_next_token_probs = log_next_token_probs - bayes_lambda * log_d_next_token_probs
                    bayes_next_token_probs_normalized = F.softmax(log_bayes_next_token_probs, dim=-1)

                    # # use probability
                    # next_token_probs = F.softmax(next_token_logits, dim=-1)
                    # d_next_token_probs = F.softmax(d_next_token_logits, dim=-1)
                    #
                    # # some element may be really small. If divide, the number will be inf.
                    # # we either make those probs 0, or add a small constant
                    # next_token_probs = next_token_probs + SMALL_CONST * (
                    #             next_token_probs <= SMALL_CONST).float().to(device)
                    # d_next_token_probs = d_next_token_probs + SMALL_CONST * (d_next_token_probs <= SMALL_CONST).float().to(device)
                    # # d_next_token_probs[d_next_token_probs <= SMALL_CONST] = 0
                    # # next_token_probs[next_token_probs <= SMALL_CONST] = 0
                    #
                    # bayes_next_token_probs = next_token_probs / d_next_token_probs
                    #
                    # # print(bayes_next_token_probs[0][:1000])
                    #
                    # # some of the elements may be 0 so that bayes_next_token_probs can be "nan"
                    # bayes_next_token_probs[bayes_next_token_probs != bayes_next_token_probs] = 0
                    # bayes_next_token_probs[bayes_next_token_probs == float("Inf")] = 0
                    #
                    # bayes_next_token_probs_normalized = bayes_next_token_probs / bayes_next_token_probs.sum(
                    #     dim=-1).unsqueeze(-1)

                    next_token = torch.multinomial(bayes_next_token_probs_normalized, num_samples=1).squeeze(1)
                else:
                    next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                if bayes and (bayes_gamma > -1 and generated_length < bayes_gamma):
                    next_token_probs = F.softmax(next_token_logits, dim=-1)
                    d_next_token_probs = F.softmax(d_next_token_logits, dim=-1)
                    bayes_next_token_probs = next_token_probs / d_next_token_probs
                    bayes_next_token_probs_normalized = bayes_next_token_probs / bayes_next_token_probs.sum(
                        dim=-1).unsqueeze(-1)
                    next_token = torch.argmax(bayes_next_token_probs_normalized, dim=-1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1)

            # # update generations and finished sentences
            # if eos_token_ids is not None:
            #     # pad finished sentences if eos_token_ids exist
            #     tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            # else:
            #     tokens_to_add = next_token
            tokens_to_add = next_token

            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            # input_ids = tokens_to_add.unsqueeze(-1)
            cur_token = tokens_to_add.unsqueeze(-1)

            # print("sample")
            # print(tokens_to_add)
            # print("max")
            # print(torch.argmax(next_token_logits, dim=-1))

            if eos_token_ids is not None:
                eos_token_id = eos_token_ids
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len + 1)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            cur_len = cur_len + 1
            generated_length += 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

    # # if there are different sentences lengths in the batch, some batches have to be padded
    # if sent_lengths.min().item() != sent_lengths.max().item():
    #     assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
    #     # finished sents are filled with pad_token
    #     decoded = input_ids.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)
    # else:
    #     decoded = input_ids
    decoded = input_ids

    for hypo_idx, hypo in enumerate(input_ids):
        decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]

    return decoded


def output_generation(prompt_text, model, input_ids, label_ids, cur_len, max_len, do_sample, temperature, top_k, top_p,
                      repetition_penalty, batch_size, tokenizer, use_domain_tag=False, use_label_tag=False,
                      domain_ids=None, bayes=False, bayes_gamma=-1, use_mlp=False, bayes_lambda=1):
    output_sequences = generate(
        model,
        input_ids=input_ids,
        label_ids=label_ids,
        cur_len=cur_len,
        max_length=max_len,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        batch_size=batch_size,
        eos_token_ids=EOS_ID,
        use_domain_tag=use_domain_tag,
        use_label_tag=use_label_tag,
        domain_ids=domain_ids,
        bayes=bayes,
        bayes_gamma=bayes_gamma,
        use_mlp=use_mlp,
        bayes_lambda=bayes_lambda,
    )

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        # print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # # Remove all text after the stop token
        # text = text[: text.find(args.stop_token) if args.stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        # total_sequence = (
        #         prompt_text + text[len(tokenizer.decode(input_ids, clean_up_tokenization_spaces=True)):]
        # )
        total_sequence = text

        generated_sequences.append(total_sequence)

    return generated_sequences


def get_gpt1_ppl(args, generated_sentence, gpt1_model, gpt1_tokenizer):
    with torch.no_grad():
        input_ids = torch.LongTensor(gpt1_tokenizer.encode(generated_sentence)).unsqueeze(0).to(args.device)
        outputs = gpt1_model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        ppl = math.exp(loss.item())
    return ppl


def get_perplexity(args, generated_sentence, gpt2_model, gpt2_tokenizer, total_added_tokens):
    # "WARNING: for baseline on simply appending token to the beginning"
    # generated_sentence_split = generated_sentence.split()[1:]
    # generated_sentence = " ".join(w for w in generated_sentence_split)

    with torch.no_grad():
        input_ids = torch.LongTensor(gpt2_tokenizer.encode(generated_sentence)).unsqueeze(0).to(args.device)
        # because the added tokens, loss will be changed. So we use lm_logits[:, :, :-ignore_token_num]
        # to calcaulate loss
        outputs = gpt2_model(input_ids, labels=input_ids, ignore_token_num=total_added_tokens)
        loss, logits = outputs[:2]
        ppl = math.exp(loss.item())
    return ppl


def get_class_prob(args, generated_sentence, eval_model, eval_tokenizer, current_class, class_idx):
    # prepare data
    # "WARNING: for baseline on simply appending token to the beginning"
    # generated_sentence_split = generated_sentence.split()[1:]
    # generated_sentence = " ".join(w for w in generated_sentence_split)

    inputs = eval_tokenizer.encode_plus(generated_sentence)
    input_ids, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs[
        "attention_mask"]
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(args.device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(args.device)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0).to(args.device)

    with torch.no_grad():
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        outputs = eval_model(**inputs)
        logits = outputs[0]
        probs = F.softmax(logits, dim=-1).squeeze(0)
        pred_prob = probs[class_idx]
        return pred_prob.item()


def get_style_prob(args, generated_sentence, style_eval_model, style_eval_tokenizer):
    inputs = style_eval_tokenizer.encode_plus(generated_sentence)
    input_ids, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs[
        "attention_mask"]
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(args.device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(args.device)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0).to(args.device)

    with torch.no_grad():
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        outputs = style_eval_model(**inputs)
        logits = outputs[0]
        probs = F.softmax(logits, dim=-1).squeeze(0)
        pred_prob = probs[0]  # 0 for movie, 1 for wiki, 2 for news
        return pred_prob.item()


def eval(args, generated_sequences, gpt2_model, gpt2_tokenizer, eval_model, eval_tokenizer, current_class, class_idx,
         total_added_tokens, original=False, gpt1_model=None, gpt1_tokenizer=None, style_eval_model=None,
         style_eval_tokenizer=None):
    all_ppl = []
    all_class_prob = []
    all_style_prob = []

    on_topic_sentence = ""
    highest_on_topic_prob = 0

    # run in loop so that we do not need to pad
    for generated_sentence in generated_sequences:
        # get perplexity
        if not original:
            if gpt1_model is not None:
                ppl = get_gpt1_ppl(args, generated_sentence, gpt1_model, gpt1_tokenizer)
            else:
                ppl = get_perplexity(args, generated_sentence, gpt2_model, gpt2_tokenizer, total_added_tokens)
            all_ppl.append(ppl)

        # # get accuracy
        # """WARNING"""
        class_prob = get_class_prob(args, generated_sentence, eval_model, eval_tokenizer, current_class, class_idx)
        # class_prob = 0
        all_class_prob.append(class_prob)
        if class_prob > highest_on_topic_prob:
            highest_on_topic_prob = class_prob
            on_topic_sentence = generated_sentence

        if style_eval_model is not None:
            style_prob = get_style_prob(args, generated_sentence, style_eval_model, style_eval_tokenizer)
            all_style_prob.append(style_prob)

    debug = False
    if debug:
        print("all_ppl")
        print(all_ppl)
        print("all_probs")
        print(all_class_prob)
        # avg_ppl = 0
        # avg_class_prob = 0

    if not original:
        avg_ppl = statistics.mean(all_ppl)
    else:
        avg_ppl = 0
    avg_class_prob = statistics.mean(all_class_prob)

    if style_eval_model is not None:
        avg_style_prob = statistics.mean(all_style_prob)
    else:
        avg_style_prob = 0

    highest_on_topic_ppl = get_perplexity(args, on_topic_sentence, gpt2_model, gpt2_tokenizer, total_added_tokens)

    return avg_ppl, avg_class_prob, "ppl: %.4f; prob: %.4f; sent: %s" % (highest_on_topic_ppl, highest_on_topic_prob,
                                                                         on_topic_sentence), all_ppl, avg_style_prob


def generate_results(args, prompt_text, model, tokenizer, total_added_tokens, eval_model, eval_tokenizer,
                     gpt1_model, gpt1_tokenizer, style_eval_model, style_eval_tokenizer):
    input_ids = torch.LongTensor(tokenizer.encode(prompt_text)).unsqueeze(0).to(args.device)

    cur_len = len(tokenizer.encode(prompt_text))
    max_len = cur_len + args.length
    do_sample = True

    # pos_label_ids = torch.LongTensor(tokenizer.encode(pos_token)).unsqueeze(0).to(args.device)
    # neg_label_ids = torch.LongTensor(tokenizer.encode(neg_token)).unsqueeze(0).to(args.device)
    pos_label_ids = torch.LongTensor(tokenizer.encode("positive")).unsqueeze(0).to(args.device)
    neg_label_ids = torch.LongTensor(tokenizer.encode("negative")).unsqueeze(0).to(args.device)
    if args.use_domain_emb:
        domain_ids = torch.LongTensor(tokenizer.encode("review")).unsqueeze(0).to(args.device)
    else:
        domain_ids = torch.LongTensor(tokenizer.encode(domain_token)).unsqueeze(0).to(args.device)

    batch_size = input_ids.shape[0]
    if args.num_return_sequences > 1:
        # Expand input to num return sequences
        input_ids = input_ids.expand(batch_size, args.num_return_sequences, cur_len)
        input_ids = input_ids.contiguous().view(
            batch_size * args.num_return_sequences, cur_len
        )  # (batch_size * num_return_sequences, cur_len)

        pos_label_ids = pos_label_ids.expand(batch_size, args.num_return_sequences, 1)
        pos_label_ids = pos_label_ids.contiguous().view(batch_size * args.num_return_sequences, 1)
        neg_label_ids = neg_label_ids.expand(batch_size, args.num_return_sequences, 1)
        neg_label_ids = neg_label_ids.contiguous().view(batch_size * args.num_return_sequences, 1)
        domain_ids = domain_ids.expand(batch_size, args.num_return_sequences, 1)
        domain_ids = domain_ids.contiguous().view(batch_size * args.num_return_sequences, 1)

        batch_size = batch_size * args.num_return_sequences

    sentiment_classes = {"positive": pos_label_ids, "negative": neg_label_ids}

    # original
    ori_generated_sequence = output_generation(prompt_text, model, input_ids, None, cur_len, max_len, do_sample,
                                               args.temperature, args.k, args.p, args.repetition_penalty, batch_size,
                                               tokenizer)

    ori_dist_1, ori_dist_2, ori_dist_3 = eval_distinct(ori_generated_sequence, tokenizer)
    all_ori_distinct = [ori_dist_1, ori_dist_2, ori_dist_3]

    if args.print_generation:
        for sent in ori_generated_sequence:
            print(sent)

    if args.do_eval:
        all_ori_sent_prob = dict()
        all_ori_style_prob = dict()
        all_ori_sent_sent = dict()
        original = False
        print("\n\n")
        for sent_idx, sent_name in enumerate(sentiment_classes.keys()):
            ppl, ori_sent_prob, ori_sent_sent, cur_ori_all_ppl, ori_style_prob = eval(args, ori_generated_sequence, model.lm_model,
                                                     tokenizer, eval_model, eval_tokenizer,
                                                     sent_name, sent_idx, total_added_tokens, original=original,
                                                     gpt1_model=gpt1_model, gpt1_tokenizer=gpt1_tokenizer,
                                                                      style_eval_model=style_eval_model,
                                                                      style_eval_tokenizer=style_eval_tokenizer)

            if sent_idx == 0:
                ori_ppl = ppl
                ori_all_ppl = cur_ori_all_ppl
            all_ori_sent_prob[sent_name] = ori_sent_prob
            all_ori_sent_sent[sent_name] = ori_sent_sent
            all_ori_style_prob[sent_name] = ori_style_prob

            original = True

    # positive + negative
    all_sent_prob = dict()
    all_style_prob = dict()
    all_sent_ppl = dict()
    all_sent_sent = dict()
    all_sent_all_ppl = dict()
    all_sent_distinct = dict()
    for sentiment, label_ids in sentiment_classes.items():
        if args.print_generation:
            print("\n\n\n")
            print("====" * 30)
        print(sentiment)
        generated_sequence = output_generation(prompt_text, model, input_ids, label_ids, cur_len, max_len,
                                               do_sample, args.temperature, args.k, args.p, args.repetition_penalty,
                                               batch_size, tokenizer, args.use_domain_tag, args.use_label_tag,
                                               domain_ids, args.bayes, args.bayes_gamma, args.use_mlp,
                                               args.bayes_lambda)
        distinct1, distinct2, distinct3 = eval_distinct(generated_sequence, tokenizer)

        # all_sent_distinct[sentiment]["1"].append(distinct1)
        # all_sent_distinct[sentiment]["2"].append(distinct2)
        # all_sent_distinct[sentiment]["3"].append(distinct3)
        all_sent_distinct[sentiment] = [distinct1, distinct2, distinct3]

        if args.print_generation:
            for sent in generated_sequence:
                print(sent)

        if args.do_eval:
            sent_idx = list(sentiment_classes.keys()).index(sentiment)
            sent_ppl, sent_prob, sent_sent, sent_all_ppl, style_prob = eval(args, generated_sequence, model.lm_model, tokenizer,
                                                  eval_model, eval_tokenizer,
                                                  sentiment, sent_idx, total_added_tokens,
                                                  gpt1_model=gpt1_model, gpt1_tokenizer=gpt1_tokenizer,
                                                                            style_eval_model=style_eval_model,
                                                                            style_eval_tokenizer=style_eval_tokenizer)
            all_sent_prob[sentiment] = sent_prob
            all_sent_sent[sentiment] = sent_sent
            all_sent_ppl[sentiment] = sent_ppl
            all_sent_all_ppl[sentiment] = sent_all_ppl
            all_style_prob[sentiment] = style_prob

    if args.do_eval:
        if args.output_results:
            full_model_name = args.load_checkpoint
            model_name = full_model_name.split("/")[-3]
            output_file = open("results_%s.txt" % model_name, "a")
            output_file.write("\n\n")
            output_file.write("*** PROMPT: {} ***\n".format(prompt_text))
            output_file.write("++++" * 30)
            output_file.write("\n")
            output_file.write("original ppl\n")
            output_file.write(str(ori_ppl))
            output_file.write("\n")
            output_file.write("original probs\n")
            output_file.write(str(all_ori_sent_prob))
            output_file.write("\n")
            output_file.write("Trained model ppl\n")
            output_file.write(str(all_sent_ppl))
            output_file.write("\n")
            output_file.write("Trained model probs")
            output_file.write(str(all_sent_prob))
            output_file.write("\n")
            output_file.write("<<<" * 30)
            output_file.write("\n")
            output_file.write("original sentiment sents\n")
            for ori_sent_class, ori_sent_sent in all_ori_sent_sent.items():
                output_file.write("=== GENERATED SEQUENCE {} ===\n".format(ori_sent_class))
                output_file.write(ori_sent_sent)
                output_file.write("\n")
            output_file.write("\n")
            output_file.write(">>>" * 30)
            output_file.write("\n")
            output_file.write("on topic sentiment sents\n")
            for sent_class, sent_sent in all_sent_sent.items():
                output_file.write("=== GENERATED SEQUENCE {} ===\n".format(sent_class))
                output_file.write(str(sent_sent))
                output_file.write("\n")
            output_file.write("\n"*10)
        else:
            print("\n\n")
            print("++++" * 30)
            print("original ppl")
            print(ori_ppl)
            print("original probs")
            print(all_ori_sent_prob)
            print("Trained model ppl")
            print(all_sent_ppl)
            print("Trained model probs")
            print(all_sent_prob)
            print()
            print("<<<" * 30)
            print("original sentiment sents")
            for ori_sent_class, ori_sent_sent in all_ori_sent_sent.items():
                print("=== GENERATED SEQUENCE {} ===".format(ori_sent_class))
                print(ori_sent_sent)
            print()
            print(">>>" * 30)
            print("on topic sentiment sents")
            for sent_class, sent_sent in all_sent_sent.items():
                print("=== GENERATED SEQUENCE {} ===".format(sent_class))
                print(sent_sent)
            print()

    print("perplexity mean and stdev")
    ori_ppl_mean = statistics.mean(ori_all_ppl)
    ori_ppl_stdev = statistics.stdev(ori_all_ppl)
    sent_all_ppl_list = []
    for c, all_ppl in all_sent_all_ppl.items():
        print(c)
        # print(all_ppl)
        c_mean = statistics.mean(all_ppl)
        c_stdev = statistics.stdev(all_ppl)
        print("attribute %s: mean: %.4f; stdev: %.4f" % (c, c_mean, c_stdev))
        sent_all_ppl_list += all_ppl
    model_ppl_mean = statistics.mean(sent_all_ppl_list)
    model_ppl_stdev = statistics.stdev(sent_all_ppl_list)
    print("original mean: %.4f, stdev: %.4f" % (ori_ppl_mean, ori_ppl_stdev))
    print("model mean: %.4f; stdev: %.4f" % (model_ppl_mean, model_ppl_stdev))

    return all_ori_sent_prob, ori_ppl, all_sent_prob, all_sent_ppl, all_ori_distinct, all_sent_distinct, \
           all_ori_style_prob, all_style_prob


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--model_type",
    #     default=None,
    #     type=str,
    #     required=True,
    #     help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    # )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--load_checkpoint", '-c', type=str, default='')

    parser.add_argument("--prompt", nargs='*', required=True)
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--bayes", action="store_true", help="use bayes for disentanglement")
    parser.add_argument("--use_mlp", action="store_true", help="use mlp to create past instead of "
                                                               "transferring past from gpt")
    parser.add_argument("--use_domain_emb", action="store_true", help="use word embedding instead of special token"
                                                                      "for domain representation")
    parser.add_argument("--bayes_gamma", type=int, default=-1, help="use bayes for the first lambda tokens")
    parser.add_argument("--bayes_lambda", type=int, default=0.1, help="weights of the domain distribution")
    parser.add_argument("--old_d_l", action="store_true", help="used to load old models")
    parser.add_argument("--use_domain_tag", action="store_true", help="condition on domain tag")
    parser.add_argument("--use_label_tag", action="store_true", help="condition on label tag")

    parser.add_argument('--do_eval', action="store_true")
    parser.add_argument('--do_style_eval', action="store_true")
    parser.add_argument('--eval_ppl_gpt1', action="store_true", help="use openai gpt to evaluate generated perplexity")
    parser.add_argument('--eval_ppl_gpt2', action="store_true", help="use gpt2 small to evaluate generated perplexity")
    parser.add_argument('--eval_ppl_gpt2_large', action="store_true", help="use gpt2 large to evaluate generated perplexity")
    parser.add_argument('--eval_model_path', type=str)
    parser.add_argument('--style_eval_model_path', type=str)
    parser.add_argument('--print_generation', action="store_true", help="print out all the generated results")
    parser.add_argument('--output_results', action="store_true", help="output results to a txt file with the "
                                                                      "loaded model_name as file name")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    lm_model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    config = GPT2Config.from_pretrained(args.model_name_or_path)
    
    add_special_tokens_(lm_model, tokenizer)
    num_added_toks = tokenizer.add_tokens([pos_token, neg_token, domain_token])
    print('We have added', num_added_toks, 'tokens for gpt2')
    lm_model.resize_token_embeddings(len(tokenizer))
    tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(args.load_checkpoint, "tokenizer"))

    total_added_tokens = len(tokenizer) - original_tokenizer_len

    model = SSTModel(lm_model, config, args.device, 0, args.bayes, args.use_mlp, args.old_d_l)
    model_weights = torch.load(os.path.join(args.load_checkpoint, "trained_model.bin"))
    model.load_state_dict(model_weights)
    model.to(args.device)
    model.eval()

    if args.old_d_l:
        model.domain_mlp.to("cpu")
        model.label_mlp.to("cpu")

    args.length = args.length
    logger.info(args)

    if args.do_eval:
        eval_tokenizer = BertTokenizer.from_pretrained(args.eval_model_path)
        eval_model = BertForSequenceClassification.from_pretrained(args.eval_model_path)
        eval_model.to(args.device)
        eval_model.eval()
    else:
        eval_model = None
        eval_tokenizer = None

    if args.do_style_eval:
        style_eval_tokenizer = BertTokenizer.from_pretrained(args.style_eval_model_path)
        style_eval_model = BertForSequenceClassification.from_pretrained(args.style_eval_model_path)
        style_eval_model.to(args.device)
        style_eval_model.eval()
    else:
        style_eval_model = None
        style_eval_tokenizer = None

    if args.eval_ppl_gpt1:
        gpt1_tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        gpt1_model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        gpt1_model.to(args.device)
    elif args.eval_ppl_gpt2:
        gpt1_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        gpt1_model = GPT2LMHeadModel.from_pretrained('gpt2')
        gpt1_model.to(args.device)
    elif args.eval_ppl_gpt2_large:
        gpt1_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        gpt1_model = GPT2LMHeadModel.from_pretrained('gpt2-large')
        gpt1_model.to(args.device)
    else:
        gpt1_tokenizer = None
        gpt1_model = None

    sentiment_classes = ["positive", "negative"]
    overall_model_on_topic = defaultdict(list)
    overall_model_ppl = defaultdict(list)
    overall_original_on_topic = defaultdict(list)
    overall_original_ppl = []
    overall_ori_distinct = []
    overall_distinct = defaultdict(list)
    overall_original_style = defaultdict(list)
    overall_model_style = defaultdict(list)
    # for c in sentiment_classes:
    #     overall_model_on_topic[c] = []
    #     overall_model_on_topic = {c: []}
    #     overall_model_ppl = {c: []}
    #     overall_original_on_topic = {c: []}
    #     overall_original_ppl = {c: []}
    for prompt_text in args.prompt:
        print("\n\n")
        print("###" * 30)
        print(prompt_text)
        ori_on_topic, ori_ppl, on_topic, ppl, all_ori_distinct, all_sent_distinct, ori_style, model_style = generate_results(args, prompt_text, model, tokenizer,
                                                                total_added_tokens, eval_model, eval_tokenizer,
                                                gpt1_model, gpt1_tokenizer, style_eval_model, style_eval_tokenizer)
        overall_original_ppl.append(ori_ppl)
        overall_ori_distinct.append(all_ori_distinct)
        for c in sentiment_classes:
            overall_original_on_topic[c].append(ori_on_topic[c])
            overall_model_on_topic[c].append(on_topic[c])
            overall_model_ppl[c].append(ppl[c])

            overall_distinct[c].append(all_sent_distinct[c])  # {"pos": [[1,2,3],[1,2,3]]}

            overall_original_style[c].append(ori_style[c])
            overall_model_style[c].append(model_style[c])

    # print(overall_model_on_topic)
    # print(overall_model_ppl)
    # print(overall_original_on_topic)
    # print(overall_original_ppl)
    print("overall_model_on_topic")
    all_model_on_topic_mean = []
    for c, l in overall_model_on_topic.items():
        c_mean = statistics.mean(l)
        all_model_on_topic_mean.append(c_mean)
        print(c, c_mean)
    print(statistics.mean(all_model_on_topic_mean))
    print()

    print("overall_model_ppl")
    all_model_ppl_mean = []
    for c, l in overall_model_ppl.items():
        c_mean = statistics.mean(l)
        all_model_ppl_mean.append(c_mean)
        print(c, c_mean)
    print(statistics.mean(all_model_ppl_mean))
    print()

    print("overall_original_on_topic")
    all_ori_on_topic_mean = []
    for c, l in overall_original_on_topic.items():
        c_mean = statistics.mean(l)
        all_ori_on_topic_mean.append(c_mean)
        print(c, c_mean)
    print(statistics.mean(all_ori_on_topic_mean))
    print()

    print("overall_original_ppl")
    print(statistics.mean(overall_original_ppl))
    print()

    print("ori distinct")
    for i in range(3):
        print(i)
        di_list = [d[i] for d in overall_ori_distinct]
        print(statistics.mean(di_list))
    print()

    print("model distinct")
    avg_distinct = defaultdict(list)
    for c, l in overall_distinct.items():
        print(c)
        for i in range(3):
            print(i)
            di_list = [d[i] for d in l]
            c_di = statistics.mean(di_list)
            print(c_di)
            avg_distinct[c].append(c_di)
    print()

    print("avg model distinct")
    for i in range(3):
        all_di = [d[i] for d in avg_distinct.values()]  # {"positive": [d1, d2, d3], "neg": [d1...]}
        print(i)
        print(statistics.mean(all_di))
    print()

    print("overall_original_style")
    all_ori_style_mean = []
    for c, l in overall_original_style.items():
        c_mean = statistics.mean(l)
        all_ori_style_mean.append(c_mean)
        print(c, c_mean)
    print(statistics.mean(all_ori_style_mean))
    print()

    print("overall_model_style")
    all_model_style_mean = []
    for c, l in overall_model_style.items():
        c_mean = statistics.mean(l)
        all_model_style_mean.append(c_mean)
        print(c, c_mean)
    print(statistics.mean(all_model_style_mean))
    print()





    # print("===="*30)
    # print("original")
    # ori_generated_sequence = output_generation(prompt_text, model, input_ids, None, cur_len, max_len, do_sample,
    #                                            args.temperature, args.k, args.p, args.repetition_penalty, batch_size,
    #                                            tokenizer)
    # if args.print_generation:
    #     for sent in ori_generated_sequence:
    #         print(sent)
    #
    # print("\n\n\n")
    # print("====" * 30)
    # print("pos")
    # output_generation(prompt_text, model, input_ids, pos_label_ids, cur_len, max_len, do_sample, args.temperature,
    #                   args.k, args.p, args.repetition_penalty, batch_size, tokenizer, args.use_domain_tag,
    #                   args.use_label_tag, domain_ids, args.bayes, args.bayes_gamma, args.use_mlp, args.bayes_lambda)
    # print("\n\n\n")
    # print("====" * 30)
    # print("neg")
    # output_generation(prompt_text, model, input_ids, neg_label_ids, cur_len, max_len, do_sample, args.temperature,
    #                   args.k, args.p, args.repetition_penalty, batch_size, tokenizer, args.use_domain_tag,
    #                   args.use_label_tag, domain_ids, args.bayes, args.bayes_gamma, args.use_mlp, args.bayes_lambda)
    #


if __name__ == "__main__":
    main()
