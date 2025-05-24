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

4. Alter the `config.yml` file to point to the correct input file path. The default is set to `data/input/aisdk-2024-05-04.csv`.

## Running the pipeline

```bash
python main.py
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











# AIS Port Detection

This project detects marine transportation ports from AIS (Automatic Identification System) data using PySpark.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download AIS Data:**
    * Go to [AIS Denmark](http://web.ais.dk/aisdata/).
    * Download a CSV data file for any specific day.
    * Unzip the file if necessary.
    * Place the CSV file into the `data/` directory. For example, `data/ais_data_20230101.csv`.
    * **Important:** Update the `AIS_DATA_FILE` variable in `main.py` to match your filename.

## Running the Project

Execute the main script:
```bash
python main.py
```

This will:
1.  Initialize a Spark session.
2.  Load and parse the AIS data.
3.  Filter noise and prepare data.
4.  Detect potential port areas using a grid-based density approach.
5.  Estimate the relative size of these ports.
6.  Generate an interactive HTML map named `detected_ports_map.html` in the project root, visualizing the ports.

## Project Structure

* `data/`: Stores the input AIS CSV data.
* `main.py`: The main script to run the analysis pipeline.
* `data_loader_and_schema.py`: Defines the schema and loads the AIS data.
* `filtering_operations.py`: Contains functions for data cleaning and filtering.
* `port_detection_logic.py`: Implements the port detection algorithm.
* `port_sizing_logic.py`: Calculates the relative sizes of detected ports.
* `visualize_ports.py`: Generates map visualizations using Folium.
* `requirements.txt`: Lists Python package dependencies.

## Port Detection Approach

1.  **Filtering:**
    * Vessels with Speed Over Ground (SOG) > 2.5 knots are filtered out.
    * Vessels with specific "at port" like Navigational Statuses (e.g., "Moored", "At Anchor") are prioritized, or if status is not helpful, low speed is the primary indicator.
    * Invalid coordinates are removed.

2.  **Detection:**
    * A grid-based density approach is used. Latitude and Longitude are discretized into grid cells.
    * Cells with a high density of stationary/slow-moving vessel signals (and/or unique vessels) are identified as potential port areas.
    * Adjacent dense cells are then grouped to form port clusters.

3.  **Sizing:**
    Relative port size is estimated based on:
    * The number of unique vessels in the port area.
    * The total number of AIS signals within the port area.
    * The spatial extent (number of grid cells forming the port).

## Customization

* **Data File:** Change `AIS_DATA_FILE` in `main.py`.
* **Filtering Parameters:** Adjust thresholds in `filtering_operations.py` (e.g., `SPEED_THRESHOLD`).
* **Port Detection Parameters:** Modify grid cell size and density thresholds in `port_detection_logic.py`.
