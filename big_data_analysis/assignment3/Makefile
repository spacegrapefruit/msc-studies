.PHONY: all up init-config init-shard1 init-shard2 add-shards install-poetry install-pip ingest filter stats clean

all: up init-config init-shard1 init-shard2 add-shards install-poetry ingest filter stats

# Bring up the Mongo sharded cluster
default: up
up:
	docker compose up -d

# Initialize the config-server replica set
init-config:
	mongosh --port 27019 --eval \
		'rs.initiate({ _id: "cfgrs", configsvr: true, members: [ { _id:0, host:"configsvr:27019" } ] })'

# Initialize shard1 replica set
init-shard1:
	mongosh --port 27018 --eval \
		'rs.initiate({ _id: "shard1", members: [ { _id:0, host:"shard1a:27018" }, { _id:1, host:"shard1b:27017" }, { _id:2, host:"shard1c:27016" } ] })'

# Initialize shard2 replica set
init-shard2:
	mongosh --port 27015 --eval \
		'rs.initiate({ _id: "shard2", members: [ { _id:0, host:"shard2a:27015" }, { _id:1, host:"shard2b:27014" }, { _id:2, host:"shard2c:27013" } ] })'

# Add shards to mongos and enable sharding
add-shards:
	mongosh --port 27020 --eval \
		'sh.addShard("shard1/shard1a:27018,shard1b:27017,shard1c:27016"); sh.addShard("shard2/shard2a:27015,shard2b:27014,shard2c:27013"); sh.enableSharding("aisdk"); sh.shardCollection("aisdk.raw", { MMSI: 1 }); sh.shardCollection("aisdk.clean", { MMSI: 1 });'

# Install Python dependencies using Poetry
install-poetry:
	poetry install --no-root -v

# Install Python dependencies using pip
install-pip:
	pip install -r requirements.txt

# Ingest data in parallel
ingest:
	poetry run python python/ingest.py

# Filter noise in parallel
filter:
	poetry run python python/filter_noise.py

# Calculate delta t and generate histogram
stats:
	poetry run python python/calculate_statistics.py

# Tear down the cluster and remove volumes
clean:
	docker compose down --volumes
