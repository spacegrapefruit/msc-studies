import os

import matplotlib.pyplot as plt
import sklearn.metrics
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS, URL, FOLDER_IN_ARCHIVE


# load the audio data
class SubsetSC(SPEECHCOMMANDS):
    def __init__(
        self,
        root,
        walker,
        class_to_idx,
        transform=None,
    ):
        base_url = "http://download.tensorflow.org/data/"
        ext_archive = ".tar.gz"

        url = os.path.join(base_url, URL + ext_archive)

        # get string representation of 'root' in case Path object is passed
        root = os.fspath(root)
        self._archive = os.path.join(root, FOLDER_IN_ARCHIVE)

        basename = os.path.basename(url)
        basename = basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(FOLDER_IN_ARCHIVE, basename)

        self._path = os.path.join(root, folder_in_archive)
        self._walker = walker
        self._transform = transform
        self._class_to_idx = class_to_idx

    def __getitem__(self, idx: int):
        waveform, *metadata = super().__getitem__(idx)

        if self._transform:
            waveform = self._transform(waveform)

        label_id = self._class_to_idx[metadata[1]]

        return waveform, label_id


def calculate_confusion_matrix(test_dataset, predictions):
    confusion_matrix = sklearn.metrics.confusion_matrix(
        [v for _, v in test_dataset],
        predictions,
    )
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix, labels):
    plt.figure(figsize=(8, 8))
    plt.imshow(confusion_matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(10), labels, rotation=45)
    plt.yticks(range(10), labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    for i in range(10):
        for j in range(10):
            plt.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="grey")

    plt.show()
