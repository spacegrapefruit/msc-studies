import argparse
import logging
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
from pesq import pesq

logging.basicConfig(level=logging.INFO)

plt.rcParams.update({"font.size": 15})

# constants
INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")
PLOT_DIR = Path("data/plots")
NOTE_NAMES = (
    ["D3", "E3", "F3", "G3", "A3", "B3"]
    + ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
)


parser = argparse.ArgumentParser(
    description="Convert an audio file to a Mel spectrogram and back to a waveform.",
)
parser.add_argument(
    "--n_fft",
    type=int,
    default=1024,
    help="Number of samples in each window of the STFT.",
)
parser.add_argument(
    "--hop_length",
    type=int,
    default=256,
    help="Number of samples between successive frames.",
)
parser.add_argument(
    "--n_mels",
    type=int,
    default=96,
    help="Number of Mel bands to generate.",
)


def draw_plots(
    waveform: np.ndarray,
    reconstructed_waveform: np.ndarray,
    mel_spectrogram: np.ndarray,
    filename_stem: str,
    sample_rate: int,
    hop_length: int,
) -> None:
    # original waveform
    plt.figure(figsize=(12, 5))
    librosa.display.waveshow(
        waveform[: 5 * sample_rate],
        sr=sample_rate,
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.ylim(-0.5, 0.5)
    plt.savefig(PLOT_DIR / f"{filename_stem}_original_waveform.png")

    # original vs reconstructed waveform
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(
        reconstructed_waveform[: 5 * sample_rate],
        sr=sample_rate,
        color="b",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.ylim(-0.5, 0.5)

    plt.subplot(2, 1, 2)
    librosa.display.waveshow(
        waveform[: 5 * sample_rate],
        sr=sample_rate,
        color="r",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.ylim(-0.5, 0.5)
    plt.savefig(PLOT_DIR / f"{filename_stem}_original_vs_reconstructed_waveform.png")

    # convert to log scale (dB) for better visualization
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Mel spectrogram of the original waveform
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(
        log_mel_spectrogram,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.savefig(PLOT_DIR / f"{filename_stem}_mel_spectrogram.png")

    # evaluate pitch (F0 contour) of the synthesized speech
    f0_original, _, _ = librosa.pyin(
        waveform,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
    )
    f0_synthesized, _, _ = librosa.pyin(
        reconstructed_waveform,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
    )

    # F0 contour (pitch variation over time)
    plt.figure(figsize=(12, 5))
    plt.plot(f0_original, label="Original F0")
    plt.plot(f0_synthesized, label="Synthesized F0", alpha=0.7)
    plt.xlabel("Frame")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.savefig(PLOT_DIR / f"{filename_stem}_f0_contour.png")

    # F0 contour (pitch variation over time) - display note names, log scale
    plt.figure(figsize=(12, 5))
    plt.plot(f0_original, label="Original F0")
    plt.plot(f0_synthesized, label="Synthesized F0", alpha=0.7)
    plt.xlabel("Frame")
    plt.ylabel("Note")
    plt.legend()
    plt.yscale("log")
    plt.yticks([librosa.note_to_hz(note) for note in NOTE_NAMES], NOTE_NAMES)
    plt.grid(which="both", axis="y", linestyle="--", alpha=0.5)
    plt.minorticks_off()
    plt.savefig(PLOT_DIR / f"{filename_stem}_f0_contour_notes.png")


def process_audio_file(
    filename: Path,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> None:
    # load an audio file
    waveform, sample_rate = librosa.load(filename, sr=None)
    logging.info(f"Duration: {len(waveform) / sample_rate:.2f} seconds")
    logging.info(f"Sample rate: {sample_rate}")

    assert sample_rate in [8000, 16000], "PESQ only supports 8kHz and 16kHz."

    # compute the STFT (magnitude)
    stft = np.abs(librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length))

    # generate a Mel filter bank
    mel_filter_bank = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)

    # apply the Mel filter bank to the power spectrogram
    mel_spectrogram = np.dot(mel_filter_bank, stft**2)

    # convert Mel spectrogram back to STFT magnitude spectrogram
    stft_magnitude = librosa.feature.inverse.mel_to_stft(
        mel_spectrogram,
        sr=sample_rate,
        n_fft=n_fft,
        power=2,
    )

    # reconstruct waveform using Griffin-Lim
    reconstructed_waveform = librosa.griffinlim(
        stft_magnitude,
        hop_length=hop_length,
        n_fft=n_fft,
        n_iter=60,
    )

    # save the reconstructed waveform as a .wav file
    reconstructed_filename = OUTPUT_DIR / filename.name
    sf.write(reconstructed_filename, reconstructed_waveform, sample_rate)

    # perform PESQ evaluation
    pesq_score = pesq(sample_rate, waveform, reconstructed_waveform)
    logging.info(f"PESQ score: {pesq_score:.2f}")

    # visualize the results
    draw_plots(
        waveform=waveform,
        reconstructed_waveform=reconstructed_waveform,
        mel_spectrogram=mel_spectrogram,
        filename_stem=filename.stem,
        sample_rate=sample_rate,
        hop_length=hop_length,
    )


if __name__ == "__main__":
    args = parser.parse_args()

    # create directories if they don't exist
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    filenames = list(INPUT_DIR.glob("*.wav"))

    # validate input files
    assert len(filenames) > 0, "No input files provided."

    # process each audio file
    for filename in filenames:
        logging.info(f"Processing {filename}")
        process_audio_file(
            filename,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
        )
        logging.info("\n")
