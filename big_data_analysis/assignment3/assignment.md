## Assignment Description

The objective of this assignment is to filter out noise from a given dataset using NoSQL databases and perform data analysis. The dataset contains vessel information, and your task is to apply various filters to eliminate noise and calculate the time difference between data points for each vessel.

## Tasks

**Task 1:** Create a NoSQL Database Cluster
* [x] Set up a cluster of NoSQL databases on your personal machine.
* [x] Configure the cluster and ensure its proper functioning, can be replication setup or sharding setup. (sharding will graded higher)
* [x] Docker compose is recommended, but not mandatory.

**Task 2:** Data Insertion in Parallel
* [x] Implement a program to read data from a CSV file.
* [x] Use separate instances of the MongoClient for each parallel thread or task.
* [x] Please insert in the database such an amount of data that is sufficient to your PC or virtual machine memory.

**Task 3:** Data Noise Filtering in Parallel
* [x] Implement a parallel data noise filtering process that operates on the inserted data.
* [ ] Identify and filter out noise based on specific criteria, including vessels with less than 100 data points and missing or invalid fields (e.g., Navigational status, MMSI, Latitude, Longitude, ROT, SOG, COG, Heading).
* [x] Store the filtered data in a separate collation within the NoSQL databases.
* [x] Consider creating appropriate indexes for efficient filtering.

**Task 4:** Calculation of Delta t and Histogram Generation
* [ ] Calculate the time difference (delta t) in milliseconds between two subsequent data points for each filtered vessel.
* [ ] Generate a histogram based on the calculated delta t values.
* [ ] Analyze the histogram to gain insights into vessel behavior.

**Task 5:** Presentation of the Solution
* [ ] Record a short video showing one Mongo instance failure (e.g., `docker kill shard1b`) and the cluster continuing to serve reads/writes.
* [ ] Upload the code and solution to the "Big data analysis" section of https://emokymai.vu.lt/.
