import torch
import torchaudio
import torch.nn as nn
import pandas as pd
import random
from torch.utils.data import Dataset


class TSED_Val(Dataset):
    def __init__(self, file_list, data_dir,
                 seg_length=10, sr=16000,
                 norm=True, mono=True,
                 **kwargs):

        self.data_dir = data_dir
        meta = pd.read_csv(file_list, sep='\t')
        meta = meta[meta['duration'] != 0]
        self.meta = meta
 
        self.seg_len = seg_length
        self.sr = sr

        self.norm = norm
        self.mono = mono

    def load_audio(self, audio_path):
        y, sr = torchaudio.load(audio_path)
        assert sr == self.sr

        # Handle stereo or mono based on self.mono
        if self.mono:
            # Convert to mono by averaging all channels
            y = torch.mean(y, dim=0, keepdim=True)
        else:
            if y.shape[0] == 1:
                pass
            elif y.shape[0] == 2:
                # Randomly pick one of the two stereo channels or take the mean
                if random.choice([True, False]):
                    y = torch.mean(y, dim=0, keepdim=True)
                else:
                    channel = random.choice([0, 1])
                    y = y[channel, :].unsqueeze(0)
            else:
                raise ValueError("Unsupported number of channels: {}".format(y.shape[0]))

        total_length = y.shape[-1]

        start = 0
        end = min(start + self.seg_len * self.sr, total_length)

        audio_clip = torch.zeros(self.seg_len * self.sr)
        audio_clip[:end - start] = y[0, start:end]

        if self.norm:
            eps = 1e-9
            max_val = torch.max(torch.abs(audio_clip))
            audio_clip = audio_clip / (max_val + eps)
        # audio_clip = self.augmenter(audio_clip)
        return audio_clip

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        audio = self.load_audio(self.data_dir + row['filename'])
        return audio, row['filename']

    def __len__(self):
        return len(self.meta)