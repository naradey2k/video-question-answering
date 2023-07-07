# Video Question Answering solution for JAS AI Hackathon

### Задача
> Анализ коротких видеофрагментов и представленные к ним вопросы по содержанию и запечатленным  событиям и/или действиям, чтобы сгенерировать наиболее подходящие ответы на русском языке.
---

### Как должно работать решение?
> Работа модели включает в себя следующие этапы: вычисление для входного видео эмбеддингов, которые далее пропускаются через MLP и Transformer адаптер для GPT декодера и декодер предсказывает ответ.
---

### Модель

**Prefix captioning**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/naradey2k/jas-ai-hackathon-solution/blob/main/videoqa_trainer.ipynb)

- модель [CLIP](https://github.com/openai/CLIP) и модель PrefixCaption [CLIPCap](https://github.com/rmokady/CLIP_prefix_caption)

### Данные

**Переведенный ActivityNet**
- [ActivityNet](https://paperswithcode.com/sota/video-question-answering-on-activitynet-qa)


### Ресурсы
**Colab PRO** с **NVIDIA Tesla V100** 

Веса модели:
[Google Drive](https://drive.google.com/file/d/16bj6OjGIbiIhGgVCO971SzORkMldEPrM/view?usp=share_link)


### Запуск демо
```python3
!pip install git+https://github.com/openai/CLIP.git
!pip install transformers, streamlit
!git clone https://github.com/naradey2k/jas-ai-hackathon-solution.git
!streamlit run Main.py
```
