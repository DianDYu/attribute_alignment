import json
import argparse
import logging
from itertools import chain
import numpy as np
import tqdm
import math
import random
import time
import pickle
import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup


ATTR_TO_SPECIAL_TOKEN = {'pad_token': '<pad>'}

# gpt-2
EOS_ID = 50256
wiki_domain_token = "<wiki>"
news_domain_token = "<news>"
dbpedia_label_tokens = ["company", "educational institution", "artist", "athlete", "officeholder",
                        "means of transportation", "building", "natural place", "village", "animal", "plant",
                        "album", "film", "written work"]
news_label_tokens = ["world", "sports", "business", "science technology"]
DATA_SPLIT_TOKEN = "<<<+++>>>"
SMALL_CONST = 1e-15


# copied from huggingface implementation
def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def reformat_data(args, topic_data, tokenizer):

    print("Building topic dataset")

    dataset = {"input_ids": [], "labels": []}

    for i, line in enumerate(topic_data):
        text, label = line.split(DATA_SPLIT_TOKEN)
        text = text.strip()
        label = label.strip()
        # dataset["input_ids"].append(tokenizer.encode(label + " " + text) + [EOS_ID])
        # get rid of " " between lable and text. Otherwise, the first token in text will be tokenized with " "

        # label = "positive" if label == pos_token else "negative"
        # if args.use_domain_emb:
        #     domain_token = "review"
        # else:
        #     domain_token = "<movie>"

        # if domain == "wiki":
        #     domain_token = wiki_domain_token
        # elif domain == "ag_news":
        #     domain_token = news_domain_token
        
        encoded_text = tokenizer.encode(label + text)[:args.max_seq_length]
        dataset["input_ids"].append(encoded_text)

        if i % 50000 == 0:
            print(i)
            

    return dataset


class MaskedNLLCriterion(nn.Module):
    def __init__(self):
        super(MaskedNLLCriterion, self).__init__()

    def forward(self, logprob, tgt, mask):
        # logprob: bze x seq_len x vocab_size
        # tgt: bze x seq_len x 1
        # mask: bze x seq_len x 1

        logprob_select = torch.gather(logprob, -1, tgt)

        # print("maskednll")
        # print(logprob_select.shape)
        # print(logprob_select)

        out = torch.masked_select(logprob_select, mask.bool())
        # print("out")
        # print(out.shape)
        # print(out.squeeze(-1))

        loss = - out.mean()  # removed masked loss in out, so we can do "mean()" here

        return loss


class TOPICDataset(Dataset):
    def __init__(self, data, input_padding_value, device="cuda"):
        self.data = data
        self.input_padding_value = input_padding_value
        self.device = device

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, index):
        data_i = self.data["input_ids"][index]
        return data_i

    def collate(self, data):
        padded_input_ids = pad_sequence([torch.tensor(d, dtype=torch.long) for d in data],
                                        batch_first=True, padding_value=self.input_padding_value).to(self.device)

        masks = (padded_input_ids != self.input_padding_value).byte()
        masks = masks.byte()

        return (padded_input_ids, masks)


class LengthSampler(Sampler):
    def __init__(self, data, batch_size, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        buckets = sorted(range(len(self.data)), key=lambda x: len(self.data[x]), reverse=True)
        batches = [buckets[i: i + self.batch_size] for i in range(0, len(buckets), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)


class TOPICModel(nn.Module):
    def __init__(self, lm_model, config, device, kl_scale=0, bayes=False, use_mlp=False, old_d_l=False,
                 use_label_only=False):
        super().__init__()
        self.lm_model = lm_model
        self.n_layers = config.n_layer
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.embed_size_per_head = int(config.n_embd / config.n_head)
        # self.past_transfer = [nn.Linear(self.n_embd, self.n_embd).to(device) for _ in range(self.n_layers)]

        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        # self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.criterion = MaskedNLLCriterion()
        self.kl_scale = kl_scale
        self.bayes = bayes
        self.use_mlp = use_mlp
        self.use_label_only = use_label_only

    def forward(
            self,
            input_ids=None,
            masks=None,
            past=None,
            attention_mask=None,
            position_ids=None):
        device = input_ids.device

        lm_logits, lm_past = self.lm_model(input_ids[:, :-1])
        lm_probs = self.logsoftmax(lm_logits)
        loss = self.criterion(lm_probs, input_ids[:, 1:].unsqueeze(-1), masks[:, 1:].unsqueeze(-1))


        print_ce_loss = loss.data.cpu().numpy()

        # assert False

        kl_loss = 0.0
        print_kl_loss = 0

        return loss, print_ce_loss, print_kl_loss


def train(args, tokenizer, model):
    gpt2_padding_value = tokenizer.convert_tokens_to_ids("<pad>")
    with open(args.train_data) as train_topic_data:
        if "formatted" in args.train_data:
            train_formatted_data = json.load(train_topic_data)
        else:
            # train_sst_data = json.load(json_file)
            train_formatted_data = reformat_data(args, train_topic_data, tokenizer)
            json_file = json.dumps(train_formatted_data)
            # f = open("formatted_" + args.train_data, "w")
            f = open(args.train_data.split(".txt")[0] + "_formatted.json", "w")
            f.write(json_file)
            f.close()
        train_dataset = TOPICDataset(train_formatted_data, gpt2_padding_value)
        if args.local_rank in [-1, 0]:
            print("number of data points: %d" % len(train_dataset))
        # train_sampler = LengthSampler(train_dataset, args.train_batch_size, shuffle=True)
        train_sampler = LengthSampler(train_dataset, args.train_batch_size, shuffle=True) \
            if args.local_rank == -1 else DistributedSamper(train_dataset)

    # with open(args.dev_data) as json_file:
    #     dev_wow_data = json.load(json_file)
    #     dev_text_data, dev_role_data, dev_knowledge_data = reformat_data(dev_wow_data)
    #     dev_dataset = WoWDataset(dev_text_data, dev_role_data, dev_knowledge_data, tokenizer)

    if args.local_rank == -1:
        train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler,
                                      collate_fn=train_dataset.collate)
    else:
        train_dataloader = DataLoader(dataset=train_dataset, sampler=train_sampler,
                                      collate_fn=train_dataset.collate, batch_size=args.train_batch_size)

    if args.local_rank in [-1, 0]:
        print("num of dataloader: %d" % len(train_dataloader))

    # dev_dataloader = DataLoader(dataset=dev_dataset, shuffle=False, batch_size=args.dev_batch_size,
    #                             collate_fn=dev_dataset.collate)

    num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_epochs
    param_optimizer = list(filter(lambda p: p[1].requires_grad, list(model.named_parameters())))
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    if args.fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    model.train()
    # model.lm_model.eval()
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model = torch.nn.DataParallel(model, list(range(args.n_gpu)))

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    model.zero_grad()

    global_steps = -1

    for ep in range(args.num_epochs):
        if args.local_rank in [-1, 0]:
            print()
            print('Epoch {}/{}'.format(ep + 1, args.num_epochs))
            print('-' * 30)
            print("trainining")
        
        # model.train()
        # model.dialog_model.eval()

        total_loss = 0
        total_ce_loss, total_kl_loss = 0, 0
        total_ppl = 0
        train_batch_print_start = time.time()

        for batch_num, train_batch in enumerate(train_dataloader):
            input_ids, masks = train_batch

            loss, print_ce_loss, print_kl_loss = model(input_ids, masks)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                print_ce_loss = print_ce_loss / args.gradient_accumulation_steps
                print_kl_loss = print_kl_loss / args.gradient_accumulation_steps

            # perplexity = np.exp(loss.item())
            total_loss += loss.item()
            total_ce_loss += print_ce_loss
            total_kl_loss += print_kl_loss
            # total_ppl += perplexity

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (batch_num + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_steps += 1

                if args.local_rank != -1:
                    total_loss = sum(all_gather_list(total_loss)) / get_world_size()
                    # total_ppl = sum(all_gather_list(total_ppl)) / get_world_size()

                if global_steps > 0 and global_steps % args.print_every == 0 and args.local_rank in [-1, 0]:
                    print(
                        "batch#: %d, global_steps: %d,  training batch loss: %.4f; lr: %.4f, "
                        "ce_loss: %.4f, kl_loss: %.4f, current batch time: %.4f " % (
                            batch_num, global_steps, total_loss / args.print_every,
                            # total_ppl / (args.print_every * args.train_batch_size),
                            scheduler.get_lr()[0] * 1000,
                            total_ce_loss / args.print_every,
                            total_kl_loss / args.print_every,
                            time.time() - train_batch_print_start)
                        )
                    train_batch_print_start = time.time()
                    total_loss = 0
                    total_ce_loss = 0
                    total_kl_loss = 0
                    # total_ppl = 0

                # if args.local_rank in [-1, 0] and ((ep + 1) % args.save_every == 0 or ep == 0):
                # Note: in the initial implementation, global steps update every gradient_accumulation_steps so it will save
                # checkpoint twice every (save_every_steps * 2) steps.

        # if args.local_rank in [-1, 0] and global_steps > 0 and global_steps % args.save_every_steps == 0:
        if args.local_rank in [-1, 0]:
            output_dir = os.path.join(args.checkpoint_dir,
                                      "checkpoint-{}".format(ep))
            # Create output directory if needed
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print("saving model to %s" % output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "trained_model.bin"))
            tokenizer_output_dir = os.path.join(output_dir, "tokenizer")
            if not os.path.exists(tokenizer_output_dir):
                os.makedirs(tokenizer_output_dir)
            tokenizer.save_pretrained(tokenizer_output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))

    #     print("Ending the current epoch")
    #     # torch.save(model.state_dict(), f"models/{ep}.pth")
    # print("Stopping after epoch %d" % ep)


def validate(args, tokenizer, model):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='gpt2-medium',
                        help='pretrained model name or path to local checkpoint')
    parser.add_argument("--load_checkpoint", '-c', type=str, default='')
    parser.add_argument("--max_seq_length", type=int, default=128)

    parser.add_argument("--kl_scale", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)

    # training gpt2 sentiment
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--dev_data", type=str)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--dev_batch_size", type=int, default=1)
    parser.add_argument("--max_history", type=int, default=3)
    parser.add_argument('--print_every', default=1, type=int)
    parser.add_argument('--save_every_steps', default=5000, type=int)
    parser.add_argument('--eval_every', default=1, type=int)
    parser.add_argument('--checkpoint_dir', type=str, default="models")
    parser.add_argument("--bayes", action="store_true", help="use bayes for disentanglement")
    parser.add_argument("--use_mlp", action="store_true", help="use mlp to create past instead of "
                                                               "transferring past from gpt")
    parser.add_argument("--use_domain_emb", action="store_true", help="use word embedding instead of special token"
                                                                      "for domain representation")
    parser.add_argument("--use_label_only", action="store_true", help="use sentiment label only as a baseline")

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument('--n_gpu', default=1, type=int)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument('--no_cuda', action='store_true')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    args.device = device

    # args.n_gpu = torch.cuda.device_count()

    # set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    config = GPT2Config.from_pretrained(args.model_name_or_path)

    lm_model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    lm_model.eval()

    # add special token
    add_special_tokens_(lm_model, tokenizer)
    num_added_toks = tokenizer.add_tokens([wiki_domain_token, news_domain_token])
    print('We have added', num_added_toks, 'tokens for gpt2')
    lm_model.resize_token_embeddings(len(tokenizer))

    model = TOPICModel(lm_model, config, args.device, args.kl_scale, args.bayes, args.use_mlp)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    # # freeze gpt2 model
    # for name, p in model.named_parameters():
    #     if "lm_model" in name:
    #         p.requires_grad = False

    for name, p in model.named_parameters():
        print(name, p.requires_grad)

    train(args, tokenizer, model)


if __name__ == '__main__':
    main()

"""CUDA_VISIBLE_DEVICES=5 python topic_finetune_training.py --train_data sst_data/topic_data/clean_agnews_train.txt --num_epochs 3 --train_batch_size 8 --gradient_accumulation_steps 2 --n_gpu 1 --fp16 --save_every_steps 3500 --checkpoint_dir models/sst_gpt2/topic_news_ft  --model_name_or_path gpt2-medium --print_every 30"""
"""CUDA_VISIBLE_DEVICES=0 nohup python topic_finetune_training.py --train_data sst_data/topic_data/clean_dbpedia_train.txt --num_epochs 2 --train_batch_size 4 --gradient_accumulation_steps 2 --n_gpu 1 --fp16 --save_every_steps 5400 --checkpoint_dir models/sst_gpt2/topic_wiki_ft  --model_name_or_path gpt2-medium --print_every 30 > topic_ft_wiki_training.log 2>&1"""