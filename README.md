# Capacitated K-Medoids Store Clustering

Capacitated K-Medoids clustering for balanced territory assignment and spatial coverage optimization.

## Project Overview

This project applies the Capacitated K-Medoids clustering algorithm to optimize BMS territory assignments by grouping stores into balanced clusters while minimizing geographical distance between stores and their assigned pool locations.

The approach incorporates capacity constraints to ensure a balanced distribution of stores across BMS areas while maintaining operational efficiency.

## Objectives

* Optimize BMS territory allocation
* Determine representative store pool locations (medoids)
* Balance store assignments across clusters
* Minimize travel distance between stores and pool locations
* Visualize cluster coverage areas spatially

## Methodology

### Data Preparation

* Store location filtering by selected branches
* Missing value handling
* Geographic coordinate validation

### Distance Calculation

* Haversine Distance Formula for calculating geographic distances between stores

### Capacitated K-Medoids Clustering

* Random medoid initialization
* Capacity-constrained store assignment
* Medoid update iteration
* Multi-run optimization to obtain the best clustering solution

### Spatial Analysis

* Convex Hull area generation
* Cluster coverage visualization
* Branch distribution evaluation

## Features

* Capacitated K-Medoids clustering
* Haversine distance calculation
* Balanced store allocation
* Medoid-based pool selection
* Cluster performance evaluation
* Interactive Folium maps
* Spatial coverage visualization

## Technologies Used

* Python
* Pandas
* NumPy
* SciPy
* Folium
* GeoPandas
* Shapely

## Outputs

* Cluster assignment results
* Pool store determination
* Distance analysis
* Interactive HTML map visualization
* Branch distribution summary

## Notes

The original dataset is not included in this repository due to confidentiality considerations. All sensitive company-related information has been removed or anonymized.
