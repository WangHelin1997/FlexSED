# FlexSED: Towards Open-Vocabulary Sound Event Detection

[![arXiv](https://img.shields.io/badge/arXiv-2409.10819-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2509.18606)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/Higobeatz/FlexSED/tree/main)


## Installation

Clone the repository:
```
git clone git@github.com:JHU-LCAP/FlexSED.git
```
Install the dependencies:
```
cd FlexSED
pip install -r requirements.txt
```


## Usage
```python
from api import FlexSED
import torch
import soundfile as sf

# load model
flexsed = FlexSED(device='cuda')

# run inference
events = ["Dog"]
preds = flexsed.run_inference("example.wav", events)

# visualize prediciton
flexsed.visualize_audio(preds, events)
```

## Training

WIP


## Reference

If you find the code useful for your research, please consider citing:

```bibtex
@article{hai2025flexsed,
  title={FlexSED: Towards Open-Vocabulary Sound Event Detection},
  author={Hai, Jiarui and Wang, Helin and Guo, Weizhe and Elhilali, Mounya},
  journal={arXiv preprint arXiv:2509.18606},
  year={2025}
}
```