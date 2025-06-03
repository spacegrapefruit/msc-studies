# AIS Port Detection

This project detects marine transportation ports from AIS (Automatic Identification System). It utilizes Apache Spark (PySpark) for efficient computation on large datasets.

## Port Detection Approach

1. **Filtering:**
   * The AIS data is loaded from a CSV file.
   * Invalid coordinates are removed.
   * Vessels with Speed Over Ground (SOG) > 2.5 knots are filtered out.
   * Vessels with Navigational Status other than "Moored" or "At anchor" are removed.
   * Duplicate entries for (MMSI, Latitude, Longitude) are deduplicated to reduce the dataset size.

2. **Detection:**
   * DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is used to detect clusters of dense points in the filtered AIS data.
   * The algorithm is configured with a distance threshold (`eps_km`) of 0.6 km and a minimum number of samples (`min_samples`) of 15.
   * Clusters with fewer than 5 unique vessels are ignored (`min_vessels_per_port` parameter).

3. **Sizing and Visualization:**
   * Each detected port cluster is evaluated for its relative size based on the number of unique vessels that have been detected within that cluster.
   * A convex hull is created around each cluster to visualize the port area.
   * A visualization map is generated using Folium, showing the following:
     - Detected port clusters with convex hulls.
     - GPS pings of vessels within the port areas.
     - Relative sizes of ports based on the number of unique vessels.
     - Background heatmap of all filtered AIS points.

## Project Structure

* `data/`: Stores the input AIS CSV data.
  * `input/`: Contains the AIS dataset files.
  * `output/`: Stores the generated visualization map.
* `python/`: Contains Python scripts for the analysis.
  * `main.py`: The main script to run the analysis pipeline.
  * `config.py`: Configuration class for loading parameters from `config.yml`.
  * `data_loader.py`: Defines the schema and loads the AIS data.
  * `filtering_operations.py`: Contains functions for data cleaning and filtering.
  * `port_detection.py`: Implements the port detection and sizing logic using DBSCAN.
  * `visualize_ports.py`: Generates map visualizations using Folium.
* `config.yml`: Configuration file for input parameters and DBSCAN settings.
* `Makefile`: Defines the build process and tasks for the project.
* `pyproject.toml`: Defines the project dependencies and configuration for Poetry.
* `README.md`: This file, providing an overview of the project.
* `requirements.txt`: Lists Python package dependencies.

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

3. Download and unzip the AIS dataset for 2024-08-08, e.g.:

   ```
   ais_2024-08-08.csv
   ```

4. Alter the `config.yml` file to point to the correct input file path. The default is set to `data/input/aisdk-2024-08-08.csv`.

## Running the pipeline

To run the AIS port detection pipeline, execute the following command in the terminal:

```bash
[poetry run] python main.py
```

This will start a Spark session, load the AIS data, filter it, detect ports, and visualize the results.

Sample output:
```
<timestamp> - INFO - Spark session started.
<timestamp> - INFO - Successfully loaded data from data/input/aisdk-2024-08-08.csv
<timestamp> - INFO - Initial row count: 20215048
<timestamp> - INFO - Starting data filtering and preparation...
<timestamp> - INFO - Rows after timestamp conversion and drop NA: 20215048
<timestamp> - INFO - Rows after filtering invalid coordinates: 20106810
<timestamp> - INFO - Rows after SOG filter (<= 2.5 knots): 8431099  
<timestamp> - INFO - Rows after filtering by Navigational Status (Moored/At anchor): 586699
<timestamp> - INFO - Rows after final selection and deduplication: 158298
<timestamp> - INFO - Collecting filtered data for DBSCAN... (count: 158298)
<timestamp> - INFO - Running DBSCAN with eps=0.6km (0.000094 radians) and min_samples=15
<timestamp> - INFO - DBSCAN complete. Found 362 clusters and 881 noise points.
<timestamp> - INFO - Skipped 314 clusters due to not meeting unique vessel threshold.
<timestamp> - INFO - Detected 48 distinct port clusters using DBSCAN.
<timestamp> - INFO - Evaluating relative port sizes...
<timestamp> - INFO - Added relative size metrics. Total ports: 48
<timestamp> - INFO - Collecting data for visualization...
<timestamp> - INFO - Collecting all filtered points for heatmap background...
<timestamp> - INFO - Generating visualization map: data/output/detected_ports_map.html
<timestamp> - INFO - Map saved to data/output/detected_ports_map.html
<timestamp> - INFO - Closing down clientserver connection
```

The generated map will be saved to `data/output/detected_ports_map.html`.

## Findings

Running on the AIS data for 2024-08-08 detected 48 distinct port clusters. The results were visualized on an interactive map, showing the detected ports, their relative sizes, locations of the GPS pings, and convex hulls around the port clusters.
