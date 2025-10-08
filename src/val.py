import torch
import os
import pandas as pd
from tqdm import tqdm
import sed_scores_eval
from desed_task.evaluation.evaluation_measures import (compute_per_intersection_macro_f1,
                                                       compute_psds_from_operating_points,
                                                       compute_psds_from_scores)
from local.utils import (batched_decode_preds,)
from utils.sed import Encoder
import numpy as np


@torch.no_grad()
def val_psds(model, val_loader, params, epoch, split, save_path, device):
    label_df = pd.read_csv(params['data'][split]['label'])
    EVENTS = label_df['label'].tolist()

    clap_emb = []
    for event in EVENTS:
        cls = torch.load(params['data']['train_data']['clap_dir'] + event + '.pt').to(device)
        cls = cls.unsqueeze(1)
        clap_emb.append(cls)
    cls = torch.cat(clap_emb, dim=1)

    encoder = Encoder(EVENTS, audio_len=10, frame_len=160, frame_hop=160, net_pooling=4, sr=16000)

    model.eval()
    test_csv = params['data'][split]["csv"]
    test_dur = params['data'][split]["dur"]

    gt = pd.read_csv(test_csv, sep='\t')

    test_scores_postprocessed_buffer = {}
    test_scores_postprocessed_buffer_tsed = {}
    test_thresholds = [0.5]
    test_psds_buffer = {k: pd.DataFrame() for k in test_thresholds}
    test_psds_buffer_tsed = {k: pd.DataFrame() for k in test_thresholds}

    for batch in tqdm(val_loader):
        audio, filenames = batch
        B = audio.shape[0]
        N = cls.shape[1]
        cls = cls.expand(B, -1, -1)

        audio = audio.to(device)
        mel = model.forward_to_spec(audio)

        preds = model(mel, cls)
        preds = torch.sigmoid(preds)
        preds = preds.reshape(B, N, -1)
        preds_tsed = preds.clone()
        # tsed assumes sound exitencance is known
        for idx, filename in enumerate(filenames):
            weak_label = list(gt[gt['filename'] == filename]['event_label'].unique())
            for j, event in enumerate(EVENTS):
                if event not in weak_label:
                    preds_tsed[idx][j] = 0.0
        # preds = preds.transpose(1, 2)

        (_, scores_postprocessed_strong, _,) = \
            batched_decode_preds(
                preds,
                filenames,
                encoder,
                median_filter=9,
                thresholds=list(test_psds_buffer.keys()), )
        test_scores_postprocessed_buffer.update(scores_postprocessed_strong)

        (_, scores_postprocessed_strong_tsed, _,) = \
            batched_decode_preds(
                preds_tsed,
                filenames,
                encoder,
                median_filter=9,
                thresholds=list(test_psds_buffer_tsed.keys()), )
        test_scores_postprocessed_buffer_tsed.update(scores_postprocessed_strong_tsed)

    ground_truth = sed_scores_eval.io.read_ground_truth_events(test_csv)
    audio_durations = sed_scores_eval.io.read_audio_durations(test_dur)

    ground_truth = {
        audio_id: ground_truth[audio_id]
        for audio_id in test_scores_postprocessed_buffer
    }
    audio_durations = {
        audio_id: audio_durations[audio_id]
        for audio_id in test_scores_postprocessed_buffer
    }

    psds1_sed_scores_eval, psds1_cls = compute_psds_from_scores(
        test_scores_postprocessed_buffer,
        ground_truth,
        audio_durations,
        dtc_threshold=0.7,
        gtc_threshold=0.7,
        cttc_threshold=None,
        alpha_ct=0.0,
        alpha_st=0.0,
        # save_dir=os.path.join(save_dir, "student", "scenario1"),
    )
    psds1_cls['overall'] = psds1_sed_scores_eval
    psds1_cls['macro_averaged'] = np.array([v for k, v in psds1_cls.items()]).mean()
    psds1_cls['name'] = 'psds1'

    psds1_sed_scores_eval_tsed, psds1_cls_tsed = compute_psds_from_scores(
        test_scores_postprocessed_buffer_tsed,
        ground_truth,
        audio_durations,
        dtc_threshold=0.7,
        gtc_threshold=0.7,
        cttc_threshold=None,
        alpha_ct=0.0,
        alpha_st=0.0,
        # save_dir=os.path.join(save_dir, "student", "scenario1"),
    )

    psds1_cls_tsed['overall'] = psds1_sed_scores_eval_tsed
    psds1_cls_tsed['macro_averaged'] = np.array([v for k, v in psds1_cls_tsed.items()]).mean()
    psds1_cls_tsed['name'] = 'psds1_tsed'

    # psds2_sed_scores_eval, psds2_cls = compute_psds_from_scores(
    #     test_scores_postprocessed_buffer,
    #     ground_truth,
    #     audio_durations,
    #     dtc_threshold=0.1,
    #     gtc_threshold=0.1,
    #     cttc_threshold=0.3,
    #     alpha_ct=0.5,
    #     alpha_st=1,
    #     # save_dir=os.path.join(save_dir, "student", "scenario1"),
    # )
    # psds2_cls['overall'] = psds2_sed_scores_eval
    # psds2_cls['macro_averaged'] = np.array([v for k, v in psds2_cls.items()]).mean()
    # psds2_cls['name'] = 'psds2'
    psds_cls = pd.DataFrame([psds1_cls, psds1_cls_tsed])
    # psds_cls = pd.DataFrame([psds1_cls, psds2_cls])
    os.makedirs(f'{save_path}/psds_cls/', exist_ok=True)
    psds_cls.to_csv(f'{save_path}/psds_cls/{epoch}.csv', index=False)

    return psds1_sed_scores_eval, psds1_sed_scores_eval_tsed