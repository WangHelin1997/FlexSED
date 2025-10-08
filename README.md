# FlexSED


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