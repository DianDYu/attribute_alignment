import os
import argparse
import logging
import re
from collections import defaultdict

import math
import statistics

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sst_generation import get_gpt1_ppl, get_class_prob, get_style_prob
from get_distinct_score import eval_distinct

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


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def read_file(args):
    all_prompt_lines = defaultdict(list)
    for filename in os.listdir(args.file_path):
        real_file_name = os.path.join(args.file_path, filename)
        with open(real_file_name) as f:
            generated_text = ""
            for line in f:
                if line.startswith("<|endoftext|>"):
                    if len(generated_text) > 0:
                        all_prompt_lines[filename].append(generated_text.strip())
                    generated_text = line[13:]
                else:
                    generated_text += line
            all_prompt_lines[filename].append(generated_text.strip())
        assert len(all_prompt_lines[filename]) == 30, "len %d  != 30" % len(all_prompt_lines[filename])
        # # debugging
        # print(all_prompt_lines[filename][:5])
        # assert False

    return all_prompt_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_ppl_gpt1', action="store_true", help="use openai gpt to evaluate generated perplexity")
    parser.add_argument('--eval_ppl_gpt2', action="store_true", help="use gpt2 small to evaluate generated perplexity")
    parser.add_argument('--eval_ppl_gpt2_large', action="store_true",
                        help="use gpt2 large to evaluate generated perplexity")
    parser.add_argument('--eval_model_path', type=str)
    parser.add_argument('--style_eval_model_path', type=str)
    parser.add_argument('--file_path', type=str)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    eval_tokenizer = BertTokenizer.from_pretrained(args.eval_model_path)
    eval_model = BertForSequenceClassification.from_pretrained(args.eval_model_path)
    eval_model.to(args.device)
    eval_model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model.to(args.device)
    
    if args.eval_ppl_gpt1:
        gpt1_tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        gpt1_model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        gpt1_model.to(args.device)
    elif args.eval_ppl_gpt2:
        gpt1_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        gpt1_model = GPT2LMHeadModel.from_pretrained('gpt2')
        gpt1_model.to(args.device)
    else:
        gpt1_tokenizer = None
        gpt1_model = None

    if len(args.style_eval_model_path) > 0:
        style_eval_tokenizer = BertTokenizer.from_pretrained(args.style_eval_model_path)
        style_eval_model = BertForSequenceClassification.from_pretrained(args.style_eval_model_path)
        style_eval_model.to(args.device)
        style_eval_model.eval()
    else:
        style_eval_model = None
        style_eval_tokenizer = None

    all_prompt_lines = read_file(args)
    print(args.file_path)

    if "neg" in args.file_path:
        class_idx = 1
    elif "pos" in args.file_path:
        class_idx = 0

    all_ppl = []
    all_class_prob = []
    all_style_prob = []

    overall_model_on_topic = []
    overall_model_style = []
    overall_model_ppl = []

    all_distinct = {"1": [], "2": [], "3": []}

    for prompt_text, generated_sequences in all_prompt_lines.items():
        on_topic_sentence = ""
        highest_on_topic_prob = 0
        for generated_sentence in generated_sequences:
            if gpt1_model is not None:
                ppl = get_gpt1_ppl(args, generated_sentence, gpt1_model, gpt1_tokenizer)
            else:
                ppl = get_gpt1_ppl(args, generated_sentence, model, tokenizer)
            all_ppl.append(ppl)

            # get accuracy
            class_prob = get_class_prob(args, generated_sentence, eval_model, eval_tokenizer, "sentiment",
                                        class_idx)
            all_class_prob.append(class_prob)
            if class_prob > highest_on_topic_prob:
                highest_on_topic_prob = class_prob
                on_topic_sentence = generated_sentence

            if style_eval_model is not None:
                style_prob = get_style_prob(args, generated_sentence, style_eval_model, style_eval_tokenizer)
                all_style_prob.append(style_prob)

        distinct1, distinct2, distinct3 = eval_distinct(generated_sequences, tokenizer)
        all_distinct["1"].append(distinct1)
        all_distinct["2"].append(distinct2)
        all_distinct["3"].append(distinct3)


        avg_ppl = statistics.mean(all_ppl)
        avg_class_prob = statistics.mean(all_class_prob)
        highest_on_topic_ppl = get_gpt1_ppl(args, on_topic_sentence, model, tokenizer)
        avg_class_style = statistics.mean(all_style_prob)

        print("ppl: %.4f; prob: %.4f; sent: %s" % (highest_on_topic_ppl, highest_on_topic_prob, on_topic_sentence))

        overall_model_on_topic.append(avg_class_prob)
        overall_model_ppl.append(avg_ppl)
        overall_model_style.append(avg_class_style)

    print("overall_model_on_topic")
    print(statistics.mean(overall_model_on_topic))
    print()

    print("overall_model_ppl")
    print(statistics.mean(overall_model_ppl))
    print()

    print("distinct")
    for i, score in all_distinct.items():
        print(i, statistics.mean(score))
    print()

    print("overall_model_style")
    print(statistics.mean(overall_model_style))


if __name__ == "__main__":
    main()
