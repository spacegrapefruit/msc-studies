# Longest Vessel Route using PySpark

This project processes Automatic Identification System (AIS) data from vessels to determine the total distance traveled by each vessel. It utilizes Apache Spark (PySpark) for efficient computation on large datasets.

The core objectives are:
- Ingest and clean AIS data from a CSV file.
- Calculate the distance between consecutive AIS pings for each vessel using the Haversine formula.
- Implement a data quality filter to identify and exclude vessels exhibiting anomalous speeds (defined as > 200 m/s between pings), which likely indicate positional errors or data corruption.
- Aggregate the segment distances to find the total route length for each valid vessel.
- Report the top 5 vessels with the longest routes on the processed day.

The script is designed to be configurable via a `config.yml` file for input parameters.

## Requirements

- Apache Spark
- Python 3.10 or 3.11
- Poetry or pip (for Python package management)

## Setup

1. Clone or copy the repo.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and unzip the AIS dataset for 2024-05-04, e.g.:

   ```
   ais_2024-05-04.csv
   ```

## Running the pipeline

```bash
python main.py /path/to/ais_2024-05-04.csv
```

It will output the MMSI and total kilometers traveled.

Sample output:
```
<timestamp> - INFO - Spark session started.
<timestamp> - INFO - Data loaded from data/input/aisdk-2024-05-04.csv, total records: 19175663
<timestamp> - INFO - Data types corrected and invalid coordinates filtered.
<timestamp> - INFO - Identifying vessels with speeds over 200 m/s...
<timestamp> - INFO - Found 277 vessels to ban due to excessive speed.
<timestamp> - INFO - Original records: 19151391. Records after removing banned vessels: 18169459.
<timestamp> - INFO - Distance segments calculated for valid vessels.
<timestamp> - INFO - Top Vessels with longest routes (excluding speed-banned vessels):
<timestamp> - INFO -   Rank 1: MMSI=219133000, Distance=793.69 km
<timestamp> - INFO -   Rank 2: MMSI=230007000, Distance=739.58 km
<timestamp> - INFO -   Rank 3: MMSI=636017000, Distance=719.01 km
<timestamp> - INFO -   Rank 4: MMSI=308803000, Distance=709.98 km
<timestamp> - INFO -   Rank 5: MMSI=244874000, Distance=685.00 km
<timestamp> - INFO - Spark session stopped.
<timestamp> - INFO - Closing down clientserver connection
```

## Findings

Running on the AIS data for 2024-05-04 shows that the vessel **MMSI=219133000** traveled **793.69 km**, which is the longest route in that 24-hour period.
