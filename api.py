import torch
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, ClapTextModelWithProjection
from src.models.transformer import Dasheng_Encoder
from src.models.sed_decoder import Decoder, TSED_Wrapper
from src.utils import load_yaml_with_includes


class FlexSED:
    def __init__(
        self,
        config_path='src/configs/model.yml',
        ckpt_path='ckpts/flexsed_as.pt',
        ckpt_url='https://huggingface.co/Higobeatz/FlexSED/resolve/main/ckpts/flexsed_as.pt',
        device='cuda'
    ):
        """
        Initialize FlexSED with model, CLAP, and tokenizer loaded once.
        If the checkpoint is not available locally, it will be downloaded automatically.
        """
        self.device = device
        params = load_yaml_with_includes(config_path)

        # Ensure checkpoint exists
        if not os.path.exists(ckpt_path):
            url = "https://huggingface.co/Higobeatz/FlexSED/resolve/main/ckpts/flexsed_as.pt"
            print(f"[FlexSED] Downloading checkpoint from {url} ...")
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
            # os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            # torch.save(state_dict, ckpt_path)  # cache locally
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")

        # Encoder + Decoder
        encoder = Dasheng_Encoder(**params['encoder']).to(self.device)
        decoder = Decoder(**params['decoder']).to(self.device)
        self.model = TSED_Wrapper(encoder, decoder, params['ft_blocks'], params['frozen_encoder'])
        self.model.load_state_dict(state_dict['model'])
        self.model.eval()

        # CLAP text model
        self.clap = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-unfused")
        self.clap.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

    @staticmethod
    def plot_and_save_event(pred_e, event_name, sr=25, out_dir="./plots"):
        """
        Save heatmap plot for a single event prediction.
        """
        os.makedirs(out_dir, exist_ok=True)
        pred_np = pred_e.squeeze(0).numpy()
        T = pred_np.shape[0]

        plt.figure(figsize=(12, 4))
        plt.imshow(
            pred_np[np.newaxis, :],
            aspect="auto",
            cmap="hot",
            extent=[0, T/sr, 0, 1]
        )
        plt.colorbar(label="Value")
        plt.xlabel("Time (s)")
        plt.ylabel("Heat")
        plt.title(f"Heatmap for {event_name}")

        save_path = os.path.join(out_dir, f"{event_name}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        return save_path

    def run_inference(self, audio_path, events, norm_audio=True):
        """
        Run inference on audio for given events.
        """
        audio, sr = librosa.load(audio_path, sr=16000)
        audio = torch.tensor([audio]).to(self.device)

        if norm_audio:
            eps = 1e-9
            max_val = torch.max(torch.abs(audio))
            audio = audio / (max_val + eps)

        clap_embeds = []
        with torch.no_grad():
            for event in events:
                text = f"The sound of {event.replace('_',' ')}"
                inputs = self.tokenizer([text], padding=True, return_tensors="pt")
                outputs = self.clap(**inputs)
                text_embeds = outputs.text_embeds.unsqueeze(1)
                clap_embeds.append(text_embeds)

            query = torch.cat(clap_embeds, dim=1).to(self.device)
            mel = self.model.forward_to_spec(audio)
            preds = self.model(mel, query)
            preds = torch.sigmoid(preds).cpu()

        return preds

    def visualize_audio(self, preds, events, out_dir="./plots"):
        results = {}
        for i, event in enumerate(events):
            pred_e = preds[i]
            plot_path = self.plot_and_save_event(pred_e, event, out_dir=out_dir)
            results[event] = {
                "prediction": pred_e.squeeze(0).tolist(),
                "plot_path": plot_path
            }
        return results


if __name__ == "__main__":
    flexsed = FlexSED(device='cuda')
    events = ["Dog"]
    preds = flexsed.run_inference("example.wav", events)
    flexsed.visualize_audio(preds, events)
