# AISDK Data Noise Filtering and Analysis

## Requirements
- Docker & Docker Compose
- Mongo Shell (`mongodb-mongosh`)
- Python 3.10 or 3.11
- Poetry or pip (for Python package management)

## Running the pipeline using Makefile

While the setup, pipeline, and clean up stages can be run manually by following the instructions below, a Makefile is provided to simplify the process. The Makefile contains targets for each step of the process, allowing you to run them with a single command:
   ```bash
   make
   ```

## Setup
1. Bring up Mongo sharded cluster:
   ```bash
   docker compose up -d
   ```

2. Initialize replica sets & sharding:

   Connect to config server (`mongosh --port 27019`):

   ```js
   // config servers
   rs.initiate({ _id: "cfgrs", configsvr: true, members: [ { _id:0, host:"configsvr:27019" } ] });
   ```
   Connect to shard1 servers (`mongosh --port 27018`):

   ```js
   // shard1 servers
   rs.initiate({ _id: "shard1", members: [ { _id:0, host:"shard1a:27018" }, { _id:1, host:"shard1b:27017" }, { _id:2, host:"shard1c:27016" } ] });
   ```
   Connect to shard2 servers (`mongosh --port 27016`):

   ```js
   // shard2 servers
   rs.initiate({ _id: "shard2", members: [ { _id:0, host:"shard2a:27015" }, { _id:1, host:"shard2b:27014" }, { _id:2, host:"shard2c:27013" } ] });
   ```
   Connect to the mongos router (`mongosh --port 27020`):

   ```js
   // mongos router
   sh.addShard("shard1/shard1a:27018,shard1b:27017,shard1c:27016");
   sh.addShard("shard2/shard2a:27015,shard2b:27014,shard2c:27013");

   // enable sharding for DB
   sh.enableSharding("aisdk");

   // enable sharding for collections
   sh.shardCollection("aisdk.raw", {MMSI:1});
   sh.shardCollection("aisdk.clean", {MMSI:1});
   ```

## Running the pipeline

1. Install Python dependencies:

   If using Poetry:

   ```bash
   poetry install --no-root -v
   ```

   If using pip:
   ```bash
   pip install -r requirements.txt
   ```

2. Ingest data in parallel:
   ```bash
   [poetry run] python python/ingest.py
   ```

3. Filter noise in parallel:
   ```bash
   [poetry run] python python/filter_noise.py
   ```

4. Calculate Δt and generate histogram:
   ```bash
   [poetry run] python python/calculate_statistics.py
   ```

## Clean up

1. Stop and remove Docker containers:
   ```bash
   docker compose down --volumes
   ```
