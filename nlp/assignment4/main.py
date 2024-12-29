import argparse
import logging
import pickle
from pathlib import Path

import nltk
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

nltk.download("gutenberg")
from nltk.corpus import gutenberg


logging.basicConfig(level=logging.INFO)

# constants
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")
PLOT_DIR = Path("data/plots")
TRAINING_TEXTS = [
    "austen-emma.txt",
    "austen-persuasion.txt",
    "austen-sense.txt",
]


parser = argparse.ArgumentParser(
    description="Text Generation using LSTM",
)
parser.add_argument(
    "--mode",
    choices=["train", "generate"],
    required=True,
    help="Mode of operation",
)
parser.add_argument(
    "--sequence_length",
    type=int,
    default=40,
    help="Length of the character sequence for training",
)
parser.add_argument(
    "--step",
    type=int,
    default=3,
    help="Step size for creating sequences",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size for training",
)
parser.add_argument(
    "--tokens_to_generate",
    type=int,
    default=200,
    help="Number of tokens to generate",
)


class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(
            self.y[idx], dtype=torch.float32
        )


class TextGenerator(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(128, output_size)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = out[:, -1, :]
        out = self.fc(out)
        return out, hidden


def train_model(
    sequence_length: int,
    step: int,
    batch_size: int,
):
    logging.info("Training the model with the following parameters:")
    logging.info(f"Sequence length: {sequence_length}")
    logging.info(f"Step size: {step}")

    text = ""
    for filename in TRAINING_TEXTS:
        text += gutenberg.raw(filename) + "\n\n\n"

    # convert to lowercase
    text = text.lower()

    chars = sorted(set(text))
    logging.info(f"Number of characters: {len(text)}")
    logging.info(f"Number of unique characters: {len(chars)}")

    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}

    sequences = []
    next_chars = []
    for i in range(0, len(text) - sequence_length, step):
        sequences.append(text[i : i + sequence_length])
        next_chars.append(text[i + sequence_length])
    logging.info(f"Number of sequences: {len(sequences)}")

    # transform inputs
    X = np.zeros((len(sequences), sequence_length, len(chars)), dtype=np.float32)
    y = np.zeros((len(sequences), len(chars)), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            X[i, t, char_to_idx[char]] = 1.0
        y[i, char_to_idx[next_chars[i]]] = 1.0

    dataset = TextDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TextGenerator(
        input_size=len(chars),
        output_size=len(chars),
    )
    model.to(TORCH_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    logging.info("Training the model...")
    for epoch in range(10):
        model.train()

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(TORCH_DEVICE)
            batch_y = batch_y.to(TORCH_DEVICE)
            optimizer.zero_grad()
            y_pred, _ = model(batch_x)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(TORCH_DEVICE)
            batch_y = batch_y.to(TORCH_DEVICE)
            y_pred, _ = model(batch_x)
            loss = criterion(y_pred, batch_y)
            total_loss += loss.item()

        logging.info(f"Epoch {epoch + 1}, Loss: {total_loss / (len(X) / batch_size)}")
    logging.info("Training complete.")

    torch.save(model, OUTPUT_DIR / "model.pth")
    logging.info(f"Model saved to {OUTPUT_DIR / 'model.pth'}")

    with open(OUTPUT_DIR / "char_maps.pkl", "wb") as f:
        pickle.dump((char_to_idx, idx_to_char), f)
    logging.info(f"Character maps saved to {OUTPUT_DIR / 'char_maps.pkl'}")


def sample(preds, diversity):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / diversity
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(
    model: torch.nn.Module,
    char_to_idx: dict,
    idx_to_char: dict,
    seed_text: str,
    diversity: float,
    tokens_to_generate: int,
) -> str:
    generated = ""
    sentence = seed_text[:40].lower()  # TODO use parameter from the model

    for _ in range(tokens_to_generate):
        x = torch.zeros(1, len(sentence), len(char_to_idx))
        for t, char in enumerate(sentence):
            x[0, t, char_to_idx[char]] = 1.0
        x = x.to(TORCH_DEVICE)

        y_pred, _ = model(x)
        preds = torch.nn.functional.softmax(y_pred.cpu(), dim=1).detach().numpy()[0]
        next_index = sample(preds, diversity)
        next_char = idx_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    return generated


if __name__ == "__main__":
    args = parser.parse_args()

    # create directories if they don't exist
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        logging.info("Training mode selected.")
        train_model(
            sequence_length=args.sequence_length,
            step=args.step,
            batch_size=args.batch_size,
        )

    elif args.mode == "generate":
        logging.info("Generate mode selected.")

        filenames = list(INPUT_DIR.glob("*.txt"))
        # validate input files
        assert len(filenames) > 0, "No input files provided."

        model = torch.load(
            OUTPUT_DIR / "model.pth",
            weights_only=False,
            map_location=torch.device(TORCH_DEVICE),
        )
        model.to(TORCH_DEVICE)
        with open(OUTPUT_DIR / "char_maps.pkl", "rb") as f:
            char_to_idx, idx_to_char = pickle.load(f)

        for filename in filenames:
            logging.info(f"Processing {filename}")
            with open(filename, "r") as f:
                seed_text = f.read()

            for diversity in [0.1, 0.5, 1.0]:
                logging.info(f"Diversity: {diversity}")
                output = generate_text(
                    model=model,
                    char_to_idx=char_to_idx,
                    idx_to_char=idx_to_char,
                    seed_text=seed_text,
                    diversity=diversity,
                    tokens_to_generate=args.tokens_to_generate,
                )

                logging.info("Seed text:")
                logging.info(seed_text[:40])
                logging.info("Generated text:")
                logging.info(output)
                logging.info("")
