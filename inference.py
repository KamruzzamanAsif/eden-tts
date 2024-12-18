"""
@author: edenmyn
@email: edenmyn
@time: 2022/10/1 13:40
@DESC:
"""

from models.edenTTS import EdenTTS
from hparams import hparams as hp
from utils.paths import Paths
import torch
import numpy as np
import librosa

import time
from utils.dsp import save_wav
from pathlib import Path
import os
from utils.log_util import get_logger
import numpy as np
from text.en_util import text_to_sequence
import argparse

from hifigan import vocoder

device = "cpu"
log = get_logger(__name__)
vocoder.to(device)


# def m_inference(tts_model, out_path, texts):
#     for text in texts:
#         log.info(f"processing text: {text}")
#         phones = text_to_sequence(text)
#         phones = torch.tensor(phones).long().unsqueeze(0).to(device)
#         s1 = time.time()
#         mel_pred = tts_model.inference(phones)
#         log.info(f"acoustic model inferance time {time.time() - s1}s")
#         log.info(f"predicted mel spectrum: {mel_pred}")
#         with torch.no_grad():
#             audio = vocoder(mel_pred.transpose(1, 2))
#         file = os.path.join(out_path, f'{text[:40]}.wav')
#         wav = audio.squeeze().cpu().detach().numpy()
#         peak = np.abs(wav).max()
#         wav = wav / peak
#         save_wav(wav, file)
#         log.info(f"Synthesized wave saved at: {file}")


def mel_to_linear(mel_spectrogram, sr=22050, n_fft=1024, n_mels=80, fmin=0.0, fmax=None):
    """
    Converts a mel-spectrogram back to a linear spectrogram.
    Args:
        mel_spectrogram (numpy.ndarray): Input mel-spectrogram (shape: [n_mels, time_steps]).
        sr (int): Sampling rate.
        n_fft (int): Number of FFT components.
        n_mels (int): Number of mel bands.
        fmin (float): Minimum frequency.
        fmax (float): Maximum frequency.
    Returns:
        numpy.ndarray: Linear spectrogram.
    """
    if fmax is None:
        fmax = sr // 2

    # Create a mel filter bank
    mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    
    # Pseudo-inverse of the mel filter bank
    mel_filter_inv = np.linalg.pinv(mel_filter)
    
    # Transpose mel_spectrogram to align time_steps with columns
    mel_spectrogram = mel_spectrogram.T  # Now shape is [time_steps, n_mels]
    
    # Convert mel-spectrogram to linear spectrogram
    linear_spectrogram = np.dot(mel_spectrogram, mel_filter_inv.T)  # Align dimensions
    linear_spectrogram = np.maximum(0.0, linear_spectrogram)  # Avoid negative values

    return linear_spectrogram.T  # Transpose back to [freq_bins, time_steps]



def m_inference(tts_model, out_path, texts):
    for text in texts:
        log.info(f"Processing text: {text}")
        
        # Convert text to phonemes
        phones = text_to_sequence(text)
        phones = torch.tensor(phones).long().unsqueeze(0).to(device)
        
        # Acoustic Model Inference
        s1 = time.time()
        mel_pred = tts_model.inference(phones)  # Mel-spectrogram prediction
        log.info(f"Acoustic model inference time: {time.time() - s1:.2f}s")
        log.info(f"Predicted mel spectrum shape: {mel_pred.shape}")
        
        # HiFi-GAN Vocoder Audio Generation
        log.info("Generating HiFi-GAN audio...")
        with torch.no_grad():
            audio_hifigan = vocoder(mel_pred.transpose(1, 2))  # HiFi-GAN expects [B, C, T]
        
        wav_hifigan = audio_hifigan.squeeze().cpu().detach().numpy()
        wav_hifigan = wav_hifigan / np.abs(wav_hifigan).max()  # Normalize
        
        # Save HiFi-GAN Output
        file_hifigan = os.path.join(out_path, f'{text[:40]}_hifigan.wav')
        save_wav(wav_hifigan, file_hifigan)
        log.info(f"Synthesized wave (HiFi-GAN) saved at: {file_hifigan}")
        
        # Griffin-Lim Audio Generation
        log.info("Generating Griffin-Lim audio...")
        mel_pred_np = mel_pred.squeeze(0).cpu().numpy()  # [n_mels, time_steps]
        try:
            spectrogram = mel_to_linear(mel_pred_np)  # Convert mel to linear spectrogram
            
            wav_griffinlim = librosa.griffinlim(
                spectrogram,
                hop_length=256,  # Ensure this matches the hop_length used in training
                win_length=1024,  # Ensure this matches the window length in training
                n_iter=60  # Number of iterations for phase reconstruction
            )
            wav_griffinlim = wav_griffinlim / np.abs(wav_griffinlim).max()  # Normalize
            
            # Save Griffin-Lim Output
            file_griffinlim = os.path.join(out_path, f'{text[:40]}_griffinlim.wav')
            save_wav(wav_griffinlim, file_griffinlim)
            log.info(f"Synthesized wave (Griffin-Lim) saved at: {file_griffinlim}")
        
        except Exception as e:
            log.error(f"Error generating Griffin-Lim audio: {e}")




def inference(texts):
    if type(texts) == str:
        texts = [texts]
    tts_model = EdenTTS(hp).to(device)
    tts_model_id = hp.tts_model_id
    paths = Paths(hp.data_path, tts_model_id)
    tts_model_path = paths.tts_latest_weights
    if not os.path.exists(tts_model_path):
        print(f"{tts_model_path} do not exist")
        return
    out_path = paths.tts_output
    os.makedirs(out_path, exist_ok=True)
    tts_model.load(tts_model_path)
    tts_model.to(device)
    m_inference(tts_model, out_path, texts)


if __name__ == "__main__":
    assert hp.speaker == "ljs"
    # set the path to the LJSpeech dataset
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--text", type=str, required=True, help="input text"
    )
    args = parser.parse_args()
    inference(args.text)



