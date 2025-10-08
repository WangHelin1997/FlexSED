import torch
import torchaudio
import torch.nn as nn
import pandas as pd
import random
from torch.utils.data import Dataset
import ast


class TSED_AS(Dataset):
    def __init__(self, data_dir, clap_dir, meta_dir, label_dir, class_list,
                 seg_length=10, sr=16000, label_sr=25, label_per_audio=[10, 10],
                 norm=True, mono=True, label_type='strong', debug=False, sample_method='random',
                 neg_removed_weight=0.25,
                 **kwargs):

        self.data_dir = data_dir
        self.clap_dir = clap_dir
        meta = pd.read_csv(meta_dir)
        meta = meta[meta['duration'] != 0]
        self.meta = meta
        if label_type == 'strong':
            label = pd.read_csv(label_dir)
            self.label = label
        else:
            self.label = None

        self.label_per_audio = label_per_audio

        self.class_list = pd.read_csv(class_list)
        self.class_dict = dict(self.class_list.set_index('id')['label'])  # Convert to dict
        # self.event_id = dict(self.class_list.set_index('label')['id'])
        self.cls_ids = sorted(self.class_list['id'].unique().tolist())
        self.sample_method = sample_method

        self.seg_len = seg_length
        self.sr = sr
        self.label_sr = label_sr
        self.label_type = label_type

        self.norm = norm
        self.mono = mono

        self.neg_removed_weight = neg_removed_weight

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

    def load_label(self, filelabel, event_label):
        target = torch.zeros(self.seg_len * self.label_sr)
        if self.label_type == 'strong':
            label = filelabel[filelabel['label'] == event_label]
            for i in range(len(label)):
                row = label.iloc[i]
                onset = row['onset']
                offset = row['offset']
                target[round(onset*self.label_sr):round(offset*self.label_sr)] = 1
        else:
            pass
        return target.unsqueeze(0)

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        audio = self.load_audio(self.data_dir + row['file_name'])

        # TBD balance positive and negative
        if self.sample_method == 'fix':
            cls_list = row['ids']
        if self.sample_method == 'random':
            cls_queue = self.cls_ids
            cls_list = random.sample(cls_queue, self.label_per_audio)
        elif self.sample_method == 'balance':
            pos_ids = ast.literal_eval(row['pos_ids'])
            neg_ids = ast.literal_eval(row['neg_ids'])
            removed_ids = ast.literal_eval(row['removed_ids'])
            N_p, N_n = self.label_per_audio
            if len(pos_ids) < N_p:
                N_n += N_p - len(pos_ids)
            assert len(neg_ids) + len(removed_ids) >= N_n
            # elif len(neg_ids) < N_n:
            #     N_p += N_n - len(neg_ids)
            sampled_pos = random.sample(pos_ids, min(N_p, len(pos_ids)))

            # Combine neg_ids and removed_ids with different sampling weights
            candidates = neg_ids + removed_ids
            weights = [1.0] * len(neg_ids) + [self.neg_removed_weight] * len(removed_ids)
            sampled_neg = random.choices(candidates, weights=weights, k=min(N_n, len(candidates)))

            cls_list = sampled_pos + sampled_neg

        cls_tokens = []
        labels = []

        filelabel = self.label[self.label['filename'] == row['file_name']]

        for cls_id in cls_list:
            event_label = self.class_dict[cls_id]
            cls = torch.load(self.clap_dir + event_label + '.pt')
            cls_tokens.append(cls)
            label = self.load_label(filelabel, event_label)
            labels.append(label)

        cls_tokens = torch.cat(cls_tokens, dim=0)
        labels = torch.cat(labels, dim=0)

        return audio, cls_tokens, labels, row['file_name']

    def __len__(self):
        return len(self.meta)