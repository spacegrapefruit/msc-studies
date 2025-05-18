## Assignment Description

Leverage the capabilities of PySpark to analyze maritime data and identify the vessel that has traveled the longest route on a specific day. This assignment will help students gain practical experience with big data technologies, particularly Apache Spark, and its Python API, PySpark.

## Dataset

The provided dataset contains Automatic Identification System (AIS) data for vessels, including details such as MMSI (Maritime Mobile Service Identity), timestamp, latitude, and longitude. Students will need to calculate the distance traveled by each vessel throughout the day and determine which vessel has the longest route.

## Tasks

* [x] Download the dataset from the given URL and unzip it to access the .csv or similar format file contained within.
* [x] Load the data into a PySpark DataFrame.
* [x] Ensure that the data types for latitude, longitude, and timestamp are appropriate for calculations and sorting.
* [x] Calculate the distance between consecutive positions for each vessel using a suitable geospatial library or custom function that can integrate
* [x] Aggregate these distances by MMSI to get the total distance traveled by each vessel on that day.
* [x] Sort or use an aggregation function to determine which vessel traveled the longest distance.
* [x] The final output should be the MMSI of the vessel that traveled the longest distance, along with the computed distance.
* [x] Ensure the code is well-documented, explaining key PySpark transformations and actions used in the process.

## Deliverables

* [x] A PySpark script that completes the task from loading to calculating and outputting the longest route.
* [x] A brief report or set of comments within the code that discusses the findings and any interesting insights about the data or the computation process.
