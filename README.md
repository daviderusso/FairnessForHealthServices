# Project Name

This repository provides a complete workflow for preprocessing healthcare service data (pharmacies, parapharmacies, hospitals), integrating it with population and territorial shapefiles, and then performing spatial kernel-based analyses to measure accessibility/fairness indicators (e.g., via Gini coefficients). The code is intended to support studies and policy decision-making regarding service coverage and equity in healthcare access.

---

## Table of Contents

1. [Overview](#overview)  
2. [Repository Structure](#repository-structure)  
3. [Requirements and Installation](#requirements-and-installation)  
4. [Data Description](#data-description)  
5. [Main Steps and Usage](#main-steps-and-usage)  
6. [Detailed File Descriptions](#detailed-file-descriptions)  
7. [Additional Notes](#additional-notes)  

---

## Overview

The project’s primary goal is to **evaluate healthcare services distribution** (like pharmacies, parapharmacies, hospitals) in relation to population data and **perform kernel-based spatial analyses** to derive metrics of accessibility fairness.

**Main features** include:
- Preprocessing CSV data (cleaning, geocoding missing addresses).
- Merging population and administrative boundary shapefiles.
- Applying convolution-like kernels on rasterized layers to measure service accessibility.
- Exporting aggregated statistics by region/province/municipality (CSV, XLSX).
- Computing fairness metrics such as the Gini coefficient.

---

## Repository Structure


- **CSV_Data_Utils.py**: Tools for CSV loading, deduplication, geocoding.  
- **Gini.py**: Classes and methods to compute Gini coefficients on kernel outputs.  
- **Kernel_Utils.py**: Functions that apply convolution kernels to raster data.  
- **Kernels.py**: Predefined kernel matrices by service type and size.  
- **SHP_Utils.py**: Helpers for GeoDataFrame handling (reprojection, merging).  
- **main.py**: Orchestrates the full workflow (preprocessing, kernel application, exports).

---

## Requirements and Installation

1. **Python 3.8+**  
2. **Dependencies** (install via pip or conda):
   - `geopandas`, `pandas`, `numpy`, `rasterio`, `shapely`, `geopy`, `pyproj`, `xlsxwriter`
3. **QGIS / GDAL** (recommended for rasterization).
4. **Data**: The repository expects CSVs (services, population), shapefiles (boundaries, grids), and raster layers.

---

## Data Description

- **services**: Raw CSVs for **pharmacies**, **parapharmacies**, **hospitals**.  
- **population**: Shapefiles (e.g., population grids), CSVs with age-based data.  
- **ImagesMap**: Raster files derived from shapefiles (via QGIS or GDAL).  
- **ImagesResults**: Output folder for kernel-convolved rasters.  
- **Results_CSV**: CSVs summarizing accessibility metrics by region/province/municipality.  
- **Results_SHP**: Shapefiles enriched with final metrics.

---

## Main Steps and Usage

1. **Clone or download** the repository.  
2. **Place input data** in `Data/` matching the subfolder structure.  
3. **Install requirements**:
   ```bash
   pip install geopandas pandas numpy rasterio shapely geopy pyproj xlsxwriter
4. **execute the file mail** python main.py
