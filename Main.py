import torch
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm, trange
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
# from torch.cuda.amp import autocast
import io
import os
import PIL
import cv2
import tempfile
import streamlit as st

from model_inference import *
from model import ClipCaptionModel

device = 'cpu'
clip_model, preprocess = clip.load("ViT-L/14@336px", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3large_based_on_gpt2')

prefix_length= 50
model_path = 'transformerfr-004.pt'
model = ClipCaptionModel('sberbank-ai/rugpt3small_based_on_gpt2', prefix_length = 50)

model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.to(device)

st.set_page_config(
    page_title="Video Analysis AI",
    page_icon="🎈",
)

def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

_max_width_()


def main():
    st.title("🤖 VideoQA")
    st.header("")

    with st.sidebar.expander("ℹ️ - О приложении", expanded=True):
        st.write(
            """
    -   *VideoQA* стремится отвечать на вопросы на естественном языке в соответствии с предоставленными видео!
    -   Модель была натренирована на переведенном на русский язык данных ActivityNet и обучена при помощи архитекур ruGPT3-large и OpenAI CLIP
    	    """
        )


    uploaded_file = st.file_uploader("📌 Загрузите видео", ['.mp4'])

    play_video = st.checkbox("Открыть видео")

    if play_video:
        video_bytes = uploaded_file.read()
        st.video(video_bytes)

    st.write("---")

    question = st.text_input("📌 Введите вопрос для видео", "")


    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    device = 'cpu'
    val_embeddings = []
    val_captions = []
    result = ''
    text = f'Question: {question}? Answer:'
    video = read_video(tfile.name, transform=None, frames_num=4)

    if len(video) > 0:
        i = image_grid(video, 2, 2)
        image = preprocess(i).unsqueeze(0).to(device)

        with torch.no_grad():
            prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)

        val_embeddings.append(prefix)
        val_captions.append(text)

    answers = []

    for i in tqdm(range(len(val_embeddings))):
        emb = val_embeddings[i]
        caption = val_captions[i]

        ans = get_ans(model, tokenizer, emb, prefix_length, caption)
        answers.append(ans['answer'])

    result = answers[0].split(' A: ')[0]

    res = st.text_input('📌 Ответ на вопрос', result, disabled=False)


if __name__ == '__main__':
    main()
