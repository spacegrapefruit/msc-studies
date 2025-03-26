# Assignment 1: AIS GPS Jamming Detection

This project analyzes AIS (Automatic Identification System) data to detect potential GPS jamming incidents. It processes large CSV datasets containing vessel tracking information, identifies abnormal movements, and applies spatial clustering to pinpoint suspicious clusters of activity.

## Overview

The tool ingests AIS data from a CSV file and processes it to detect anomalies in vessel movement. Two versions of the processing pipeline are provided:

- **Multiprocessing Version (`main.py`):**  
  Utilizes Python's multiprocessing module to handle data in parallel, splitting the workload across several worker processes.

- **Sequential Version (`main_sequential.py`):**  
  Processes the CSV file sequentially in chunks. This simpler version is implemented for comparison and debugging purposes.

## Dependencies

- **Python Version:** 3.10.* (other versions may work, but are untested)
- **Python Standard Library:**  
  `argparse`, `math`, `time`, `logging`, `multiprocessing` (for `main.py`), `collections`, `dataclasses`
- **Third-Party Libraries:**
  - [NumPy](https://numpy.org/)
  - [Pandas](https://pandas.pydata.org/)
  - [scikit-learn](https://scikit-learn.org/) (for DBSCAN clustering)

All dependencies are managed using [Poetry](https://python-poetry.org/).

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/spacegrapefruit/msc-studies.git
   cd msc-studies/big_data_analysis/assignment1
   ```

2. **Install Dependencies using Poetry:**
   ```bash
   poetry install -vvv
   ```

3. **Activate the Virtual Environment (Optional):**
   Poetry automatically creates and manages a virtual environment for you. To activate it, run:
   ```bash
   poetry shell
   ```

## Usage

Both versions accept similar command-line arguments for configuration.

### Common Command-Line Arguments

- **`--file-path`**  
  *Description:* Path to the AIS CSV file.  
  *Default:* `data/aisdk-2025-02-24.csv`

- **`--log-level`**  
  *Description:* Logging level for the application.  
  *Choices:* `DEBUG`, `INFO`, `WARNING`, `ERROR`  
  *Default:* `INFO`

- **`--chunk-size`**  
  *Description:* Number of rows to process per chunk.  
  *Default:* `100000`

- **`--worker-processes`** (only for `main.py`)  
  *Description:* Number of worker processes to spawn.  
  *Default:* `max(mp.cpu_count() - 1, 1)`

### Running the Multiprocessing Version

Use the following command:

```bash
poetry run python main.py \
    --file-path data/aisdk-2025-02-24.csv \
    --log-level INFO \
    --worker-processes 4 \
    --chunk-size 100000
```

### Running the Sequential Version

Use the following command:

```bash
poetry run python main_sequential.py \
    --file-path data/aisdk-2025-02-24.csv \
    --log-level INFO \
    --chunk-size 100000
```

## Implementation Details

### Multiprocessing Version (`main.py`)

- **Reader Process:**  
  Reads the CSV file in chunks, distributes data to worker processes via task queues.
- **Worker Processes:**  
  Each worker processes its assigned chunk, detects anomalies, and sends results back via a result queue.
- **Main Function:**  
  Orchestrates the reader and worker processes, collects alerts, performs time-sliced clustering, and logs the results.

### Sequential Version (`main_sequential.py`)

- **Sequential Worker Function:**  
  Processes each CSV chunk in a linear flow, updating vessel states and collecting alerts.
- **Main Function:**  
  Iterates through CSV chunks sequentially, performs anomaly detection and clustering, and logs the findings.
