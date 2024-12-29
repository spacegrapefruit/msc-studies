import torch
import torchaudio

from torchaudio.transforms import MFCC


# pad waveform to target length
def pad_waveform(waveform, target_length):
    if waveform.size(1) < target_length:
        padding = target_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif waveform.size(1) > target_length:
        waveform = waveform[:, :target_length]

    return waveform


# input length: 16000
mfcc_transform = MFCC(
    sample_rate=16000,
    n_mfcc=20,
    melkwargs={
        "n_fft": 512,  # 32ms
        "hop_length": 128,  # 8ms
        "n_mels": 64,
        "f_max": 4000,
    },
)


def transform_waveform(waveform):
    return mfcc_transform(pad_waveform(waveform, 16000))


def transform_waveform_40db(waveform):
    waveform = mfcc_transform(pad_waveform(waveform, 16000))

    signal_power = waveform.pow(2).mean().item()
    snr_db = 40
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(waveform) * (noise_power ** 0.5)  # white noise
    waveform += noise

    return waveform


def transform_waveform_20db(waveform):
    waveform = mfcc_transform(pad_waveform(waveform, 16000))

    signal_power = waveform.pow(2).mean().item()
    snr_db = 20
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(waveform) * (noise_power ** 0.5)  # white noise
    waveform += noise

    return waveform
