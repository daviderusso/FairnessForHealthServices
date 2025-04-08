import math
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

import CSV_Data_Utils as csv
import Kernel_Utils
import Kernels
import SHP_Utils as shp


def pharmacy_preprocess(path):
    """
    Preprocesses pharmacy data from a CSV file.

    The function reads a CSV file with pharmacy data (using a global 'extension' variable),
    clears duplicated rows based on the pharmacy name field, checks and updates coordinates
    using a specific function for pharmacies and parapharmacies, then splits the resulting
    data into valid and erroneous rows (those with missing coordinates). Both results are
    saved as separate CSV files with suffixes "_valid" and "_errors".

    Args:
        path (str): The file path (without extension) where the CSV is located.

    Returns:
        None
    """
    # Read CSV data with specified encoding and separator
    df_pharmacy = pd.read_csv(path + extension, encoding='ISO-8859-1', sep=';', low_memory=False)
    # Remove duplicate rows based on the pharmacy name field
    df_pharmacy = csv.clear_duplicated_row(df_pharmacy, csv.name_data_pharmacy)
    # Update coordinates using the defined function (handles errors and cleansing)
    df_pharmacy_new = csv.check_and_update_coordinates_parapharm_pharm(df_pharmacy, csv.latitude, csv.longitude,
                                                                       csv.address_data_pharmacy,
                                                                       csv.name_data_pharmacy)
    # Filter out rows with invalid/missing coordinates
    df_pharmacy_valid = csv.drop_location_with_no_coordinates(df_pharmacy_new)
    df_pharmacy_valid.to_csv(path + "_valid" + extension, index=False, sep=';')
    # Extract rows with erroneous coordinates
    df_pharmacy_not_valid = csv.extract_location_with_no_coordinates(df_pharmacy_new)
    df_pharmacy_not_valid.to_csv(path + "_errors" + extension, index=False, sep=';')


def parapharmacy_preprocess(path):
    """
    Preprocesses parapharmacy data from a CSV file.

    This function mirrors pharmacy_preprocess but is dedicated to parapharmacies.
    It reads the parapharmacy CSV, removes duplicates, checks and updates coordinates,
    then saves valid rows and rows with errors into separate CSV files.

    Args:
        path (str): The file path (without extension) where the CSV is located.

    Returns:
        None
    """
    df_parapharmacy = pd.read_csv(path + extension, encoding='ISO-8859-1', sep=';', low_memory=False)
    df_parapharmacy = csv.clear_duplicated_row(df_parapharmacy, csv.name_data_parapharmacy)
    df_parapharmacy_new = csv.check_and_update_coordinates_parapharm_pharm(df_parapharmacy, csv.latitude,
                                                                           csv.longitude,
                                                                           csv.address_data_parapharmacy,
                                                                           csv.name_data_parapharmacy)
    df_parapharmacy_valid = csv.drop_location_with_no_coordinates(df_parapharmacy_new)
    df_parapharmacy_valid.to_csv(path + "_valid" + extension, index=False, sep=';')
    df_parapharmacy_not_valid = csv.extract_location_with_no_coordinates(df_parapharmacy_new)
    df_parapharmacy_not_valid.to_csv(path + "_errors" + extension, index=False, sep=';')


def hospital_preprocess(path):
    """
    Preprocesses hospital data from a CSV file.

    The function reads hospital data from a CSV file, removes duplicate rows based on the hospital name,
    retrieves and updates coordinate values using a dedicated hospital geolocation function,
    then filters valid records and erroneous ones (with missing coordinates) saving them as separate files.

    Args:
        path (str): The file path (without extension) where the CSV is located.

    Returns:
        None
    """
    df_hospital = pd.read_csv(path + extension, encoding='ISO-8859-1', sep=';', low_memory=False)
    df_hospital = csv.clear_duplicated_row(df_hospital, csv.name_data_hospital)
    df_hospital_new = csv.get_coordinates_hospital(df_hospital, csv.latitude, csv.longitude,
                                                   csv.address_data_hospital, csv.name_data_hospital)
    df_hospital_valid = csv.drop_location_with_no_coordinates(df_hospital_new)
    df_hospital_valid.to_csv(path + "_valid" + extension, index=False, sep=';')
    df_hospital_not_valid = csv.extract_location_with_no_coordinates(df_hospital_new)
    df_hospital_not_valid.to_csv(path + "_errors" + extension, index=False, sep=';')


def add_services_data_in_shp(gdf, data_label, services):
    """
    Merges service count data into a GeoDataFrame based on point location.

    The function reprojects the GeoDataFrame to the target coordinate system, adds new columns
    (as specified by data_label), and iterates over each service provided in the 'services' dictionary.
    For each service entry (with latitude and longitude), it checks the location against the shapefile:
      - First, by checking if the point lies within any geometry.
      - If not found, it attempts switching latitude and longitude.
      - As a fallback, it finds the nearest geometry.
    The service count is then incremented in the corresponding geometry record.

    Args:
        gdf (geopandas.GeoDataFrame): The GeoDataFrame representing spatial data (e.g., a grid).
        data_label (list or str): Column name(s) to add to count the services.
        services (dict): A dictionary of service DataFrames keyed by service name.

    Returns:
        geopandas.GeoDataFrame: The updated GeoDataFrame with service count data.
    """
    # Ensure the GeoDataFrame is in the target coordinate system
    gdf = shp.reproject_shp(gdf)
    # Add new column(s) for the service data
    gdf = shp.add_column_data_to_shp(gdf, data_label)

    # Loop through each service and update the corresponding count in the shapefile
    for s in services:
        print(s)
        curr_df = services[s]
        for index, row in curr_df.iterrows():
            lat = row[csv.latitude]
            long = row[csv.longitude]
            shp_id = shp.check_point_in_shp(gdf, lat, long)
            if shp_id != -1:
                # Increment the count in the identified geometry cell
                gdf.at[shp_id, s] = gdf.at[shp_id, s] + 1
            else:
                # Try switching latitude and longitude
                shp_id = shp.check_point_in_shp(gdf, long, lat)
                if shp_id != -1:
                    print("SWITCHING LAT-LONG")
                    gdf.at[shp_id, s] = gdf.at[shp_id, s] + 1
                else:
                    # Get the nearest geometry cell if no direct match is found
                    shp_id = shp.get_nearest_shp(gdf, lat, long)
                    if shp_id != -1:
                        print("GET NEAREST")
                        gdf.at[shp_id, s] = gdf.at[shp_id, s] + 1
                    else:
                        print("NOT FOUND: " + str(index) + " - " + str(lat) + " - " + str(long))
    print("DONE")
    return gdf


def add_national_information_in_shp(gdf, gdf_national_boundaries, labels, output_path):
    """
    Merges national boundary attribute data into a GeoDataFrame and writes output shapefiles.

    For each label provided, the function merges data from a corresponding national boundary
    GeoDataFrame (using spatial intersection) into the primary GeoDataFrame. The updated GeoDataFrame
    is saved as a new shapefile for every label.

    Args:
        gdf (geopandas.GeoDataFrame): The primary GeoDataFrame to update.
        gdf_national_boundaries (dict): A dictionary of GeoDataFrames keyed by boundary label.
        labels (list): A list of labels representing different national boundaries.
        output_path (str): The file path where the new shapefiles will be saved.

    Returns:
        geopandas.GeoDataFrame: The updated GeoDataFrame after merging national data.
    """
    gdf_in_progress = gdf
    for index in range(len(labels)):
        print(labels[index])
        gdf_in_progress = shp.check_shp_in_shp_and_merge_data(gdf_national_boundaries[labels[index]], gdf_in_progress,
                                                              labels[index])
        # Write updated GeoDataFrame as shapefile for the given boundary label
        gdf_in_progress.to_file(output_path + "_" + labels[index] + ".shp")
        print(labels[index] + " DONE")

    return gdf_in_progress


def remove_shp_with_empty_field(gdf, labels):
    """
    Removes rows from a GeoDataFrame where specified fields are null.

    For each field listed in 'labels', rows with null values in that field are filtered out.

    Args:
        gdf (geopandas.GeoDataFrame): The GeoDataFrame to filter.
        labels (list): A list of column names to check for null values.

    Returns:
        geopandas.GeoDataFrame: The filtered GeoDataFrame with no null values in specified fields.
    """
    for index in range(len(labels)):
        gdf = gdf[~gdf[labels[index]].isnull()]
        print(labels[index] + " DONE")
    return gdf


def preprocess_population_data(pop_data, age_ranges):
    """
    Preprocesses population data for analysis.

    The function performs several cleaning steps:
      - Drops unneeded columns.
      - Filters rows for total values in gender and marital status.
      - Groups data by territory and age and aggregates values.
      - Cleans and converts age data.
      - Creates age range bins based on specified age_ranges.
      - Computes the percentage of the population for each age range within each territory.

    Args:
        pop_data (pandas.DataFrame): The raw population DataFrame.
        age_ranges (list): Bin edges for the age ranges.

    Returns:
        pandas.DataFrame: The preprocessed population data with age range percentages.
    """
    # Drop columns that are not needed
    columns_to_drop = ['ITTER107', 'TIPO_DATO15', 'Tipo di indicatore demografico', 'SEXISTAT1', 'ETA1', 'STATCIV2',
                       'TIME', 'Seleziona periodo', 'Flag Codes', 'Flags']
    pop_data.drop(columns_to_drop, axis=1, inplace=True)

    # Filter rows for total gender and marital status values
    pop_data = pop_data[pop_data['Sesso'] == 'totale']
    pop_data = pop_data[pop_data['Stato civile'] == 'totale']

    # Drop columns used solely for filtering
    pop_data.drop(['Sesso', 'Stato civile'], axis=1, inplace=True)

    # Group data by territory and age, summing the 'Value'
    pop_data = pop_data.groupby(['Territorio', 'Età'])['Value'].sum().reset_index()

    # Remove rows where age is 'totale' and clean the age string
    pop_data = pop_data[pop_data['Età'] != 'totale']
    pop_data['Età'] = pop_data['Età'].str.replace(' anni', '')
    pop_data['Età'] = pop_data['Età'].str.replace(' e più', '')
    pop_data['Età'] = pop_data['Età'].astype(int)

    # Bin age data into ranges specified by age_ranges
    pop_data['range_età'] = pd.cut(pop_data['Età'], bins=age_ranges, right=False)
    pop_data = pop_data.groupby(['Territorio', 'range_età']).sum().reset_index()
    pop_data.drop(columns=['Età'], inplace=True)

    # Compute the percentage of the population within each age range per territory
    grouped_sum = pop_data.groupby('Territorio')['Value'].transform('sum')
    pop_data['perc_pop_range'] = round((pop_data['Value'] / grouped_sum) * 100.0, 2)
    return pop_data


def merge_pop_data_in_shp(gdf, pop_data):
    """
    Merges population data into a GeoDataFrame based on a common territory field.

    The function creates new columns in the GeoDataFrame for each unique age range from the population data.
    For each record (e.g., a municipality) in the GeoDataFrame, it finds corresponding population data and calculates
    a weighted population value based on the percentage for each age range and total population (assumed in the column 'TOT_P_2021').

    Args:
        gdf (geopandas.GeoDataFrame): The GeoDataFrame with spatial boundaries.
        pop_data (pandas.DataFrame): The preprocessed population data.

    Returns:
        geopandas.GeoDataFrame: The updated GeoDataFrame with new population-based columns.
    """
    l = len(gdf)
    new_col = pd.unique(pop_data['range_età'])
    # Create new columns with prefix "P" for each age range
    for c in new_col:
        gdf["P" + c] = 0.0

    # For each geometry in the GeoDataFrame, merge corresponding population values
    for index, row in gdf.iterrows():
        comune = row['COMUNE']
        result = pop_data[pop_data['Territorio'] == comune]
        for ind, r in result.iterrows():
            c = "P" + result['range_età'][ind]
            p = (result['perc_pop_range'][ind] * row['TOT_P_2021']) / 100.0
            if math.isnan(p):
                gdf.at[index, c] = 0.0
            else:
                gdf.at[index, c] = p

        print(str((index * 100.0) / l))
    print("DONE")
    return gdf


def kernel_list_application_normalized(column_list_services, column_list_population, sizes, img_data_services,
                                       img_data_population, meta_data, path):
    """
    Applies normalized population-weighted kernels to service images for various sizes.

    For each service and population pair and for every specified kernel size, the function applies a kernel
    using a normalized population weighting. The result is written to a TIFF file using rasterio.

    Args:
        column_list_services (list): List of service identifiers.
        column_list_population (list): List of population data identifiers.
        sizes (list): List of kernel sizes.
        img_data_services (dict): Dictionary mapping service identifiers to their raster image data.
        img_data_population (dict): Dictionary mapping population identifiers to their raster image data.
        meta_data (dict): Metadata for writing the output TIFF files.
        path (str): Output file directory/path.

    Returns:
        None
    """
    for r in column_list_services:
        for p in column_list_population:
            for s in sizes:
                curr_application = Kernel_Utils.apply_kernel_normalized_pop_weight(
                    img_data_services[r],
                    Kernels.Kernel[r][s],
                    img_data_population[p]
                )
                meta = meta_data[p]
                output_filename = path + "img_results_norm_pop_" + r + "_" + p + "_" + str(s) + ".tif"
                with rasterio.open(output_filename, 'w', **meta) as dst:
                    dst.write(curr_application, 1)
                print("DONE: " + r + "_" + p + "_" + str(s))


def kernel_application_normalized(column_list_services, column_list_population, kernelSizes, img_data_services,
                                  img_data_population, meta_data, path):
    """
    Applies normalized population-weighted kernels to service images using a specified kernel size per service.

    For each service, the kernel size is derived from the kernelSizes dictionary.
    Then, for each population data type, the normalized population-weighted kernel is applied and saved as a TIFF file.

    Args:
        column_list_services (list): List of service identifiers.
        column_list_population (list): List of population data identifiers.
        kernelSizes (dict): Dictionary mapping service identifiers to kernel sizes.
        img_data_services (dict): Dictionary mapping service identifiers to their raster image data.
        img_data_population (dict): Dictionary mapping population identifiers to their raster image data.
        meta_data (dict): Metadata for output TIFF files.
        path (str): Output file directory/path.

    Returns:
        None
    """
    for r in column_list_services:
        s = kernelSizes[r]
        for p in column_list_population:
            print(p)
            curr_application = Kernel_Utils.apply_kernel_normalized_pop_weight(
                img_data_services[r],
                Kernels.Kernel[r][s],
                img_data_population[p]
            )
            meta = meta_data[p]
            output_filename = path + "img_results_norm_pop_" + r + "_" + p + "_" + str(s) + ".tif"
            with rasterio.open(output_filename, 'w', **meta) as dst:
                dst.write(curr_application, 1)
            print("DONE: " + r + "_" + p + "_" + str(s))


def kernel_application_normalized_no_pop(column_list_services, sizes, img_data_services,
                                         meta_data, path):
    """
    Applies kernels to service images without population weighting.

    For every service and each specified kernel size, the kernel is applied without using population data.
    The results are written to TIFF files.

    Args:
        column_list_services (list): List of service identifiers.
        sizes (list): List of kernel sizes.
        img_data_services (dict): Dictionary mapping service identifiers to their raster image data.
        meta_data (dict): Metadata for output TIFF files.
        path (str): Output file directory/path.

    Returns:
        None
    """
    for r in column_list_services:
        for s in sizes:
            curr_application = Kernel_Utils.apply_kernel_no_weight(
                img_data_services[r],
                Kernels.Kernel[r][s]
            )
            meta = meta_data[r]
            output_filename = path + "img_results_norm_nopop_" + r + "_" + str(s) + ".tif"
            with rasterio.open(output_filename, 'w', **meta) as dst:
                dst.write(curr_application, 1)
            print("DONE: " + r + "_" + str(s))


def kernel_application(column_list_services, column_list_population, sizes, img_data_services, img_data_population,
                       meta_data, path):
    """
    Applies population-weighted kernels to service images.

    For each service, population data pair, and kernel size, the function applies the kernel
    (without normalization) using the respective population data. The resulting images are saved as TIFF files.

    Args:
        column_list_services (list): List of service identifiers.
        column_list_population (list): List of population data identifiers.
        sizes (list): List of kernel sizes.
        img_data_services (dict): Dictionary mapping service identifiers to their raster image data.
        img_data_population (dict): Dictionary mapping population identifiers to their raster image data.
        meta_data (dict): Metadata for output TIFF files.
        path (str): Output file directory/path.

    Returns:
        None
    """
    for r in column_list_services:
        for p in column_list_population:
            for s in sizes:
                curr_application = Kernel_Utils.apply_kernel_with_pop_weight(
                    img_data_services[r],
                    Kernels.Kernel[r][s],
                    img_data_population[p]
                )
                meta = meta_data[p]
                output_filename = path + "img_results_pop_" + r + "_" + p + "_" + str(s) + ".tif"
                with rasterio.open(output_filename, 'w', **meta) as dst:
                    dst.write(curr_application, 1)


def export_kernel_results_csv(sizes, column_list_services, column_list_population, temp_file_name, extension, labels,
                              gdf_national_boundaries):
    """
    Exports aggregated kernel result statistics (total and average) to CSV files.

    For each specified kernel size, and for each combination of service and population data,
    the function reads the corresponding TIFF file, crops the image based on each national boundary geometry,
    computes the total sum and mean pixel value, and writes these statistics into a CSV file.

    Args:
        sizes (list): List of kernel sizes.
        column_list_services (list): List of service identifiers.
        column_list_population (list): List of population data identifiers.
        temp_file_name (str): Base name of the temporary TIFF files.
        extension (str): File extension of the TIFF files.
        labels (list): List of boundary labels.
        gdf_national_boundaries (dict): Dictionary mapping boundary labels to GeoDataFrames.

    Returns:
        None
    """
    for s in sizes:
        df_column_s = []
        df_column_s.append("LOCATION")
        for r in column_list_services:
            for p in column_list_population:
                df_column_s.append("TOT_" + r + "_" + p)
                df_column_s.append("AVG_" + r + "_" + p)

        df = pd.DataFrame(columns=df_column_s)
        for r in column_list_services:
            for p in column_list_population:
                tot_val = []
                avg_val = []
                name_val = []
                filename = temp_file_name + r + "_" + p + "_" + str(s) + extension
                for l in labels:
                    with rasterio.open(filename) as src:
                        # For each boundary, crop image by geometry and compute statistics
                        for index, row in gdf_national_boundaries[l].iterrows():
                            out_image, out_transform = mask(src, [mapping(row.geometry)], crop=True)
                            tot_val.append(out_image.sum())
                            avg_val.append(out_image.mean())
                            name_val.append(row[l])
                df["LOCATION"] = name_val
                df["TOT_" + r + "_" + p] = tot_val
                df["AVG_" + r + "_" + p] = avg_val

        df.to_csv(base_path + csv_res + "Results_" + str(s) + ".csv", index=False)
        print('DONE SIZE ', s)


def export_kernel_results_csv_no_pop(sizes, column_list_services, temp_file_name, extension, labels,
                                     gdf_national_boundaries):
    """
    Exports kernel result statistics to CSV files for the case without population data.

    For each kernel size and for each service, the function reads the corresponding TIFF file,
    crops the image according to national boundary geometries, computes total and average values,
    and writes the statistics into a CSV file for each boundary.

    Args:
        sizes (list): List of kernel sizes.
        column_list_services (list): List of service identifiers.
        temp_file_name (str): Base name of the temporary TIFF files.
        extension (str): File extension of the TIFF files.
        labels (list): List of boundary labels.
        gdf_national_boundaries (dict): Dictionary mapping boundary labels to GeoDataFrames.

    Returns:
        None
    """
    for s in sizes:
        global_df_column_s = []
        global_df_column_s.append("LOCATION")
        for r in column_list_services:
            global_df_column_s.append("TOT_" + r)
            global_df_column_s.append("AVG_" + r)

        for l in labels:
            local_df = pd.DataFrame(columns=global_df_column_s)
            for r in column_list_services:
                filename = temp_file_name + r + "_" + str(s) + extension
                tot_val = []
                avg_val = []
                name_val = []
                with rasterio.open(filename) as src:
                    for index, row in gdf_national_boundaries[l].iterrows():
                        out_image, out_transform = mask(src, [mapping(row.geometry)], crop=True)
                        tot_val.append(out_image.sum())
                        avg_val.append(out_image.mean())
                        name_val.append(row[l])
                local_df["LOCATION"] = name_val
                local_df["TOT_" + r] = tot_val
                local_df["AVG_" + r] = avg_val

            local_df.to_csv(base_path + csv_res + "Results_" + l + "_" + str(s) + ".csv", index=False)
        print('DONE SIZE ', s)


def export_kernel_results_csv_with_pop_common_size(sizes, column_list_services, column_list_population, temp_file_name,
                                                   extension, labels, gdf_national_boundaries):
    """
    Exports kernel result statistics with population data (using a common kernel size) to CSV files.

    For each kernel size and for each boundary, the function aggregates statistics for every service and
    population combination from the corresponding TIFF file and writes the data into a CSV file.

    Args:
        sizes (list): List of kernel sizes.
        column_list_services (list): List of service identifiers.
        column_list_population (list): List of population data identifiers.
        temp_file_name (str): Base name of the temporary TIFF files.
        extension (str): File extension of the TIFF files.
        labels (list): List of boundary labels.
        gdf_national_boundaries (dict): Dictionary mapping boundary labels to GeoDataFrames.

    Returns:
        None
    """
    for s in sizes:
        global_df_column_s = []
        global_df_column_s.append("LOCATION")
        for r in column_list_services:
            for p in column_list_population:
                global_df_column_s.append("TOT_" + r + "_" + p)
                global_df_column_s.append("AVG_" + r + "_" + p)

        for l in labels:
            local_df = pd.DataFrame(columns=global_df_column_s)
            for r in column_list_services:
                for p in column_list_population:
                    filename = temp_file_name + r + "_" + p + "_" + str(s) + extension
                    tot_val = []
                    avg_val = []
                    name_val = []
                    with rasterio.open(filename) as src:
                        for index, row in gdf_national_boundaries[l].iterrows():
                            out_image, out_transform = mask(src, [mapping(row.geometry)], crop=True)
                            tot_val.append(out_image.sum())
                            avg_val.append(out_image.mean())
                            name_val.append(row[l])
                    local_df["LOCATION"] = name_val
                    local_df["TOT_" + r + "_" + p] = tot_val
                    local_df["AVG_" + r + "_" + p] = avg_val

            local_df.to_csv(base_path + csv_res + "Results_" + l + "_" + str(s) + ".csv", index=False)
        print('DONE SIZE ', s)


def export_kernel_results_csv_with_pop(kernel_size, column_list_services, column_list_population, temp_file_name,
                                       extension, labels, gdf_national_boundaries):
    """
    Exports kernel result statistics with population data to CSV files using a service-specific kernel size.

    For each service, the corresponding kernel size is extracted from kernel_size. For each population column,
    the function reads the associated TIFF file, computes total and average values over national boundaries,
    and writes the results into a CSV file.

    Args:
        kernel_size (dict): Dictionary mapping service identifiers to a kernel size.
        column_list_services (list): List of service identifiers.
        column_list_population (list): List of population data identifiers.
        temp_file_name (str): Base name of the temporary TIFF files.
        extension (str): File extension of the TIFF files.
        labels (list): List of boundary labels.
        gdf_national_boundaries (dict): Dictionary mapping boundary labels to GeoDataFrames.

    Returns:
        None
    """
    for r in column_list_services:
        s = kernel_size[r]
        global_df_column_s = []
        global_df_column_s.append("LOCATION")
        for p in column_list_population:
            global_df_column_s.append("TOT_" + r + "_" + p)
            global_df_column_s.append("AVG_" + r + "_" + p)

        for l in labels:
            local_df = pd.DataFrame(columns=global_df_column_s)
            for p in column_list_population:
                filename = temp_file_name + r + "_" + p + "_" + str(s) + extension
                tot_val = []
                avg_val = []
                name_val = []
                with rasterio.open(filename) as src:
                    for index, row in gdf_national_boundaries[l].iterrows():
                        out_image, out_transform = mask(src, [mapping(row.geometry)], crop=True)
                        tot_val.append(out_image.sum())
                        avg_val.append(out_image.mean())
                        name_val.append(row[l])
                local_df["LOCATION"] = name_val
                local_df["TOT_" + r + "_" + p] = tot_val
                local_df["AVG_" + r + "_" + p] = avg_val

            local_df.to_csv(base_path + csv_res + "Results_" + l + "_" + r + ".csv", index=False)


def merge_csv_to_xlsx(csv_files, sheetName, path, output_file):
    """
    Merges multiple CSV files into a single Excel file with separate sheets.

    For each CSV file in the list, the function reads the file into a DataFrame and writes it to a separate
    sheet in the resulting Excel file using XlsxWriter.

    Args:
        csv_files (list): List of CSV file names (paths are concatenated with 'path').
        sheetName (list): List of sheet names corresponding to each CSV file.
        path (str): The directory path where the CSV files are located.
        output_file (str): The output Excel file name.

    Returns:
        None
    """
    with pd.ExcelWriter(path + output_file, engine='xlsxwriter') as writer:
        for i, csv_file in enumerate(csv_files):
            df = pd.read_csv(path + csv_file)
            df.to_excel(writer, sheet_name=f'{sheetName[i]}', index=False)


def add_results_data_in_shp(gdf, image, data_label):
    """
    Adds result data from a raster image to a GeoDataFrame.

    The function creates a new column in the GeoDataFrame (named by data_label) and, for each geometry,
    extracts pixel values by masking the raster image with the geometry. The mean pixel value is computed
    and assigned to the corresponding row.

    Args:
        gdf (geopandas.GeoDataFrame): The GeoDataFrame with geometries.
        image (rasterio Dataset): The open raster image.
        data_label (str): The column name to store the computed pixel value.

    Returns:
        geopandas.GeoDataFrame: The updated GeoDataFrame with the new column filled.
    """
    # Create new column with initial value 0
    gdf[data_label] = 0
    pixel_values = []
    for idx, shape in gdf.iterrows():
        geometry = shape.geometry
        # Mask the image to extract pixels within the geometry
        masked_image, _ = mask(image, [geometry], crop=True, nodata=0)
        # Compute the average pixel value
        pixel_value = masked_image.mean()
        pixel_values.append(pixel_value)
    # Update the GeoDataFrame with the computed pixel values
    gdf[data_label] = pixel_values
    print("DONE " + data_label)
    return gdf


def gini_calculator(impact):
    """
    Calculates the Gini coefficient for a given series of impact values.

    The function sorts the impact values, computes the cumulative impact, normalizes it,
    and then calculates the Gini coefficient using the formula:
        Gini = 1 - (2 * (total_relative_impact / (n - 1)))

    Args:
        impact (pandas.Series): A series of numerical impact values.

    Returns:
        float: The computed Gini coefficient.
    """
    sorted_impact = impact.sort_values(ascending=True)
    total = np.sum(sorted_impact)  # Total sum of impact values
    cum_impact = np.cumsum(sorted_impact)  # Cumulative sum of impact values
    relative_impact = cum_impact / total  # Normalized cumulative impact
    total_relative_impact = np.sum(relative_impact[:-1])
    gini = 1 - (2 * (total_relative_impact / (len(relative_impact) - 1)))
    return gini


def compute_gini_national_bound(gdf, labels, fields, pop_field):
    """
    Computes and prints the Gini coefficient for specified fields within national boundaries.

    For each boundary (based on the provided labels) and for each field, the function calculates the Gini coefficient
    on the subset of the GeoDataFrame where the population field is non-zero. Results are printed and written to an output file.

    Args:
        gdf (geopandas.GeoDataFrame): The GeoDataFrame containing the impact data.
        labels (list): A list of column names representing different national boundaries.
        fields (list): A list of field names for which the Gini coefficient is computed.
        pop_field (str): The column name representing the population field.

    Returns:
        None
    """
    results = []
    for l in labels:
        boundary = set(gdf[l])
        for b in boundary:
            impact = gdf.loc[(gdf[pop_field] != 0.0) & (gdf[l] == b)]
            for f in fields:
                gini = gini_calculator(impact[f])
                res = b + " \t " + f + "\t " + str(gini)
                print(res)
                results.append(res)
            results.append("")
        print("---------------------------")
        print("---------------------------")
        print("---------------------------")

    with open("output.txt", 'w') as file:
        for item in results:
            file.write(f"{item}\n")


if __name__ == '__main__':
    # path base
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Data")

    # folder
    servicesCSV = "services"
    population = "population"
    images_map = "ImagesMap"
    images_res = "ImagesResults"
    csv_res = "Results_CSV"
    shp_res = "Results_SHP"
    shp_urb = "RESULTS_URBANIZATION_DEGREE"

    # File extensions e suffissi
    extension = ".csv"
    valid = "_valid"
    errors = "_errors"
    extension_img = ".tif"

    # Percorsi servizi
    pharmacy_path = os.path.join(base_path, servicesCSV, "PHARMACY")
    parapharmacy_path = os.path.join(base_path, servicesCSV, "PARAPHARMACY")
    hospital_path = os.path.join(base_path, servicesCSV, "HOSPITAL")

    # Percorsi shapefile popolazione
    shapefile_path = os.path.join(base_path, population, "GrigliaPop2021_ITA_DatiProv.shp")
    shapefile_with_bounds = os.path.join(base_path, population, "Population_with_bounds.shp")
    shapefile_with_bounds_cleaned = os.path.join(base_path, population, "Population_with_bounds_cleaned.shp")
    shapefile_pop_perc = os.path.join(base_path, population, "Population_pop_perc.shp")
    shapefile_final = os.path.join(base_path, population, "Population_final.shp")

    # Percorsi shapefile risultati
    shapefile_results_nopop = os.path.join(base_path, shp_res, "ResultsSHP_noPop.shp")
    shapefile_results_pop = os.path.join(base_path, shp_res, "ResultsSHP_Pop.shp")
    shapefile_results_pop_normalized = os.path.join(base_path, shp_res, "ResultsSHP_Pop_Norm.shp")
    shapefile_results_pop_final = os.path.join(base_path, shp_res, "ResultsSHP_Pop_Final.shp")
    shapefile_results_pop_urb = os.path.join(base_path, shp_urb, "ResultsSHP_Pop_Urb.shp")

    # Limiti amministrativi
    shapefile_comune = os.path.join(base_path, "LimitiComuniProvince", "Com01012023", "Com01012023_WGS84.shp")
    shapefile_provincia = os.path.join(base_path, "LimitiComuniProvince", "ProvCM01012023", "ProvCM01012023_WGS84.shp")
    shapefile_regione = os.path.join(base_path, "LimitiComuniProvince", "Reg01012023", "Reg01012023_WGS84.shp")

    # Dati popolazione
    population_data_2021_0 = os.path.join(base_path, population, "POPOLAZIONE_COMUNE_ETA_2021_0_40.csv")
    population_data_2021_1 = os.path.join(base_path, population, "POPOLAZIONE_COMUNE_ETA_2021_41_60.csv")
    population_data_2021_2 = os.path.join(base_path, population, "POPOLAZIONE_COMUNE_ETA_2021_61_100.csv")
    population_data_2021_processed = os.path.join(base_path, population, "POPOLAZIONE_COMUNE_ETA_2021_processed.csv")

    ####################### CSV PREPROCESS
    # NOTE: IF COORDINATES GIVES ERRORS ARE REMOVED
    # some street does not return any coordinate because are not present in open street map.
    print("PARAPHARMACY")
    parapharmacy_preprocess(parapharmacy_path)
    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")
    print("PHARMACY")
    pharmacy_preprocess(pharmacy_path)
    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")
    print("HOSPITAL")
    hospital_preprocess(hospital_path)
    print("-----------------------------------------------------------------------------")

    ####################### SHP PREPROCESS
    # ADD DATA FORM NATIONAL BOUNDARY IN GRID_MATRIX SHP POPULATION
    # READ DATA ABOUT REGIONE - PROVINCIA - COMUNE
    labels = ['DEN_REG', 'DEN_UTS', 'COMUNE']
    gdf_national_boundaries = {
        labels[0]: shp.reproject_shp(gpd.read_file(shapefile_regione)),
        labels[1]: shp.reproject_shp(gpd.read_file(shapefile_provincia)),
        labels[2]: shp.reproject_shp(gpd.read_file(shapefile_comune)),
    }
    gdf = shp.reproject_shp(gpd.read_file(shapefile_path))
    gdf_with_national_bounds = add_national_information_in_shp(gdf, gdf_national_boundaries, labels,
                                                               shapefile_with_bounds)
    gdf_with_national_bounds.to_file(shapefile_with_bounds)

    # REMOVE INVALID SHP
    gdf = shp.reproject_shp(gpd.read_file(shapefile_with_bounds))
    gdf_cleaned = remove_shp_with_empty_field(gdf, labels)
    gdf_cleaned.to_file(shapefile_with_bounds_cleaned)

    # COMPUTE POPULATION_BY_AGE
    age_ranges = [0, 18, 65, 150]
    pop_data2021_0 = pd.read_csv(population_data_2021_0, low_memory=False)
    pop_data2021_1 = pd.read_csv(population_data_2021_1, low_memory=False)
    pop_data2021_2 = pd.read_csv(population_data_2021_2, low_memory=False)
    pop_data2021 = pd.concat([pop_data2021_0, pop_data2021_1, pop_data2021_2], ignore_index=True)
    pop_data2021 = preprocess_population_data(pop_data2021, age_ranges)
    pop_data2021.to_csv(population_data_2021_processed)

    # COMPUTE POPULATION_BY_AGE
    pop_data2021 = pd.read_csv(population_data_2021_processed, low_memory=False)
    gdf = gpd.read_file(shapefile_with_bounds_cleaned)
    gdf = merge_pop_data_in_shp(gdf, pop_data2021)
    gdf.to_file(shapefile_pop_perc)

    # IMPORT SERVICES DATA
    # FOR COORDINATES THAT GIVES NOT FOUND IN EXISTING SHAPE, WE GET THE SHP CLOSEST TO THE COORDINATES
    data_label = ['PHARMACY', 'PARAPHARM', 'HOSPITAL']
    services = {
        data_label[0]: pd.read_csv(pharmacy_path + valid + extension, encoding='ISO-8859-1', sep=';', low_memory=False),
        data_label[1]: pd.read_csv(parapharmacy_path + valid + extension, encoding='ISO-8859-1', sep=';',
                                   low_memory=False),
        data_label[2]: pd.read_csv(hospital_path + valid + extension, encoding='ISO-8859-1', sep=';', low_memory=False),
    }

    gdf = shp.reproject_shp(gpd.read_file(shapefile_pop_perc))
    gdf = add_services_data_in_shp(gdf, data_label,
                                   services)
    gdf.to_file(shapefile_final)

    ####################### FAIRNESS PREPROCESS
    ####### MANUAL STEP
    # GENERATE IMAGES - USING QGIS FOLLOWING PROCEDURE
    # column_list = ['TOT_P_2021', 'PHARMACY', 'PARAPHARMACY', 'HOSPITAL', 'P[0,18)', 'P[18,65)', 'P[65,150)']
    # gdf = gpd.read_file(shapefile_final)
    # minx, miny, maxx, maxy = gdf.total_bounds
    # width_km = get_distance(miny, minx, miny, maxx)
    # height_km = get_distance(miny, minx, maxy, minx)
    # print("W: " + str(width_km))
    # print("H: " + str(height_km))
    # 1- Open shapefile_final in qgis
    # 2- select Raster -> Conversion -> Rasterize (Vector to Raster)
    # 3- Use the following paramether:
    #   3.1 - Input Layer = shapefile_final
    #   3.2 - Field Used for burn-in value = each element of column_list
    #   3.3 - Output raster size units = pixel
    #   3.4 - Width/Horizontal resolution = width_km = 1079.5545591302741
    #   3.5 - Height/Vertical resolution = height_km = 1289.5561857469352
    #   3.6 - Output extent = Calculate From Layer -> shapefile_final
    #   3.6 - specified nodata value = 0
    # 4- save the new generated image
    # the gdal command is: gdal_rasterize -l Population_final -a TOT_P_2021 -ts 1080.0 1290.0 -a_nodata 0.0 -te 6.623841985 35.48371291 18.527293766 47.095056627 -ot Float32 -of GTiff "/mnt/DATA/Ricerca/12-ProgettoPNRR Age-It/Code/Walkability/Data/population/Population_final.shp" /tmp/processing_ftakqL/0aaf23b31b9b48b9bcc446b148160f26/OUTPUT.tif

    # APPLY KERNELS WITH POP WEIGHT
    img_name = "img_"
    column_list_population = ['P[0,18)', 'P[18,65)', 'P[65,150)']
    column_list_services = ['PHARMACY', 'PARAPHARMACY', 'HOSPITAL']
    kernelSizes = {
        "PHARMACY": 21,
        "PARAPHARMACY": 5,
        "HOSPITAL": 121,
    }
    img_data_population = dict()
    meta_data = dict()
    for c in column_list_population:
        filename = os.path.join(base_path, images_map, img_name + c + extension_img)
        try:
            with rasterio.open(filename) as src:
                img_data_population[c] = src.read(1)
                meta_data[c] = src.meta
        except Exception as e:
            print(f"Error reading image {filename}: {e}")
    img_data_services = dict()
    for c in column_list_services:
        filename = os.path.join(base_path, images_map, img_name + c + extension_img)
        try:
            with rasterio.open(filename) as src:
                img_data_services[c] = src.read(1)
        except Exception as e:
            print(f"Error reading image {filename}: {e}")
    kernel_application_normalized(column_list_services, column_list_population, kernelSizes, img_data_services,
                                  img_data_population, meta_data, os.path.join(base_path, images_res))

    # CREATE CSV FROM KERNEL APPLICATION IMAGES WITH POP
    # a DF for each size, and boundary - DF will be saved in csv format
    # EACH DF will be composed as follow: on column the service name and pop, on the row the location name
    img_name = "img_results_norm_pop_"
    column_list_services = ['PHARMACY', 'PARAPHARMACY', 'HOSPITAL']
    kernelSizes = {
        "PHARMACY": 21,
        "PARAPHARMACY": 5,
        "HOSPITAL": 121,
    }
    column_list_population = ['P[0,18)', 'P[18,65)', 'P[65,150)']
    labels = ['DEN_REG', 'DEN_UTS', 'COMUNE']
    gdf_national_boundaries = {
        labels[0]: shp.reproject_shp(gpd.read_file(shapefile_regione)),
        labels[1]: shp.reproject_shp(gpd.read_file(shapefile_provincia)),
        labels[2]: shp.reproject_shp(gpd.read_file(shapefile_comune)),
    }
    export_kernel_results_csv_with_pop(kernelSizes, column_list_services, column_list_population,
                                       os.path.join(base_path, images_res, img_name), extension_img, labels,
                                       gdf_national_boundaries)

    ## MERGE CSV IN A SINGLE CSV FOR EACH NATIONAL BOUNDARY
    boundary_list = ['DEN_REG', 'DEN_UTS', 'COMUNE']
    column_list_services = ['PHARMACY', 'PARAPHARMACY', 'HOSPITAL']
    for bl in boundary_list:
        csv_files = []
        sheetName = []
        for s in column_list_services:
            csv_files.append('Results_' + bl + '_' + str(s) + '.csv')
            sheetName.append(str(s))
        merge_csv_to_xlsx(csv_files, sheetName, os.path.join(base_path, csv_res), 'Results_' + bl + '.xlsx')

    # IMPORT RESULTS DATA IN SHP WITH POP
    gdf = shp.reproject_shp(gpd.read_file(shapefile_final))
    data_label = ['PHARMACY', 'PARAPHARMACY', 'HOSPITAL']
    data_label = ['PHARMACY']
    column_list_population = ['P[0,18)']
    column_list_population = ['P[0,18)', 'P[18,65)', 'P[65,150)']
    kernelSizes = {
        "PHARMACY": 21,
        "PARAPHARMACY": 5,
        "HOSPITAL": 121,
    }
    for l in data_label:
        for p in column_list_population:
            s = kernelSizes[l]
            with rasterio.open(os.path.join(base_path, images_res, 'img_results_norm_pop_' + l + '_' + p + '_' + str(
                    s) + '.tif')) as current_image:
                image_data = current_image.read(1)  # Read pixel values
                image_transform = current_image.transform  # Get affine transform
                p1 = p.replace('P', '')
                l1 = "_"
                if l == "HOSPITAL":
                    l1 = "H"
                elif l == "PHARMACY":
                    l1 = "F"
                elif l == "PARAPHARMACY":
                    l1 = "P"
                data_string = p1 + l1
                gdf = add_results_data_in_shp(gdf, current_image, data_string)
    gdf.to_file(shapefile_results_pop)

    # NORMALIZE RESULTS
    gdf = shp.reproject_shp(gpd.read_file(shapefile_results_pop))
    field_to_normalize = ['[0,18)F', '[18,65)F', '[65,150)F',
                          '[0,18)P', '[18,65)P', '[65,150)P',
                          '[0,18)H', '[18,65)H', '[65,150)H']
    for k in field_to_normalize:
        print(k)
        min_value = gdf[k].min()
        max_value = gdf[k].max()
        gdf['N' + k] = (gdf[k] - min_value) / (max_value - min_value)

    gdf.to_file(shapefile_results_pop_normalized)

    # CREATE FAIRNESS DISTRIBUTION INDEX
    gdf = shp.reproject_shp(gpd.read_file(shapefile_results_pop_normalized))
    age = ['[0,18)', '[18,65)', '[65,150)']
    service = ['F', 'P', 'H']

    w = [1.0, 1.0, 1.0]
    for a in age:
        gdf['TN' + a] = [0.0] * len(gdf['N' + a + 'F'])

    for a in age:
        print(a)
        for i in range(0, len(service)):
            print(service[i])
            gdf['TN' + a] = gdf['TN' + a] + (gdf['N' + a + service[i]] * w[i])

    gdf.to_file(shapefile_results_pop_final)

    gdf = shp.reproject_shp(gpd.read_file(shapefile_results_pop_final))
    pop_field = "TOT_P_2021"
    labels = ['DEN_REG', 'DEN_UTS', 'COMUNE']
    field = [
        'N[65,150)F',
        'N[65,150)P',
        'N[65,150)H',
        'TN[65,150)'
    ]
    compute_gini_national_bound(gdf, labels, field, pop_field)
