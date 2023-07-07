import os
import pickle
import sys
import argparse
import json
import io
import cv2
import os
import PIL
import random
import clip
import numpy as np
import pandas as pd
import torch
import torchvision
import transformers
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn

from enum import Enum
from PIL import Image
from torch.nn import functional as nnf
from torch.utils.data import Dataset
from typing import Tuple, Optional, Union
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm, trange

def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt='',
        embed=None,
        entry_count=1,
        entry_length=67,
        top_p=0.98,
        temperature=1,
        stop_token='.',
):

    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():
        for entry_idx in range(entry_count):
            if not tokens:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)

            emb_tokens = model.gpt.transformer.wte(tokens)

            if embed is not None:
                generated = torch.cat((embed, emb_tokens), dim=1)
            else:
                generated = emb_tokens

            for i in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                top_k = 2000
                top_p = 0.98

                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)

                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)

                generated = torch.cat((generated, next_token_embed), dim=1)

                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())

            output_text = tokenizer.decode(output_list)
            output_text = filter_ngrams(output_text)
            generated_list.append(output_text)

    return generated_list[0]

def filter_ngrams(output_text):
    a_pos = output_text.find(' A:')
    sec_a_pos = output_text.find(' A:', a_pos + 1)

    return output_text[:sec_a_pos]

def image_grid(imgs, rows, cols):
    pils = imgs

    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid


def read_video(path, transform=None, frames_num=9, window=30):
    frames = []

    cap = cv2.VideoCapture(path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    N = length // (frames_num)
    current_frame = 1

    for i in range(length):
        ret, frame = cap.read(current_frame)

        if ret and i == current_frame and len(frames) < frames_num:
            size = 193, 193
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame.thumbnail(size, Image.ANTIALIAS)

            frames.append(frame)
            current_frame += N

    cap.release()

    return frames

def get_caption(model, tokenizer, prefix, prefix_length, prompt=''):
    device = 'cpu'
    prefix = prefix.to(device)

    with torch.no_grad():
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)

        if prompt:
            generated_text_prefix = generate2(model, tokenizer, prompt=prompt, embed=prefix_embed)
        else:
            generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)

    return generated_text_prefix.replace('\n', ' ')


def get_ans(model, tokenizer, clip_emb, prefix_length, prompt):
    output = get_caption(model, tokenizer, clip_emb, prefix_length, prompt=prompt)
    ans = output[len(prompt):].strip()

    return {'answer': ans}


def main(config):
    prefix_length = config['prefix_len']
    device = device = torch.device('gpu') if torch.cuda.is_available() else torch.device('cpu')

    clip_model, preprocess = clip.load("ViT-L-14-336px.pt", device=device, jit=False)
    clip_model.to(device)

    tokenizer = GPT2Tokenizer.from_pretrained(config['gpt'])

    model_path = config['model']
    model = ClipCaptionPrefix(gpt=config['gpt'], prefix_length=prefix_length)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)

    out_path = 'Features_val.pkl'

    val_embeddings = []
    val_captions = []
    device = 'cuda'

    for q, p in zip(df_eval.question, df_eval.paths):
        text = f'Question:{q}? Answer:'
        path = f'{config["video_path"]}{p}.mp4'

        try:
            video = read_video(path, transform=None, frames_num=4) # 4
            if len(video) > 0:
                i = image_grid(video, 2, 2) # 2 2
                image = preprocess(i).unsqueeze(0).to(device)
                with torch.no_grad():
                    prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
                    # prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
                val_embeddings.append(prefix)
                val_captions.append(text)
        except Exception as e:
            print(e)

    answers = []

    for i in tqdm(range(len(val_embeddings))):
        emb = val_embeddings[i]
        caption = val_captions[i]

        ans = get_ans(model, tokenizer, emb, prefix_length, caption)
        answers.append(ans['answer'])

    df = pd.DataFrame({'answer': answers})
    df.to_csv(os.path.join(config['output_path'], 'answer.csv'))


def infer():
    WEIGHTS_PATH = 'transformerfr-004.pt'

    conf = dict(
        model=WEIGHTS_PATH,
        # video_path=args.video_path,
        gpt='/app/rugptsmall',
        prefix_len=50
    )

    device = torch.device('gpu') if torch.cuda.is_available() else torch.device('cpu')

    main(conf)
