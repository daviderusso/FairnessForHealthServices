from shapely.geometry import Point
from pyproj import CRS
import geopandas as gpd
import pandas as pd

# Define the target coordinate reference system as EPSG:4326
target_system = 4326


def reproject_shp(gdf):
    """
    Reprojects a GeoDataFrame to the target coordinate reference system (EPSG:4326).

    If the GeoDataFrame's CRS does not match the target system, it is reprojected.
    Otherwise, the original GeoDataFrame is returned.

    Args:
        gdf (geopandas.GeoDataFrame): The GeoDataFrame to reproject.

    Returns:
        geopandas.GeoDataFrame: The reprojected GeoDataFrame.
    """
    gdf_reprojected = gdf
    if gdf.crs != target_system:
        target_crs = CRS.from_epsg(target_system)
        gdf_reprojected = gdf.to_crs(target_crs)

    return gdf_reprojected


def check_point_in_shp(gdf, lat, long):
    """
    Checks whether a point specified by latitude and longitude is contained within any geometry of a GeoDataFrame.

    The function cleans the latitude and longitude if they are strings (by replacing commas,
    quotes, etc.) and attempts to convert them to float. If conversion fails, it returns -1.
    It also adjusts the values if they appear to be off by a magnitude (e.g., if greater than 10,000).
    A Shapely Point is then created, and the function returns the index of the first geometry
    that contains the point, or -1 if none is found.

    Args:
        gdf (geopandas.GeoDataFrame): The GeoDataFrame containing geometries.
        lat (str or float): The latitude of the point.
        long (str or float): The longitude of the point.

    Returns:
        int: The index of the geometry that contains the point, or -1 if no match is found.
    """
    if isinstance(lat, str):
        lat.replace(",", ".")
        lat.replace("'", "")
        lat.replace('"', "")
        try:
            lat = float(lat)
        except ValueError:
            return -1
    if isinstance(long, str):
        long.replace(",", ".")
        long.replace("'", "")
        long.replace('"', "")
        try:
            long = float(long)
        except ValueError:
            return -1
    # Adjust if latitude and longitude are of the wrong magnitude
    if lat > 10000:
        lat = lat / 1000
    if long > 10000:
        long = long / 1000
    # Create a point with (longitude, latitude)
    point = Point(long, lat)
    # Check which geometries contain the point
    contains_point = gdf.geometry.contains(point)
    index = contains_point.idxmax() if contains_point.any() else -1
    return index


def get_nearest_shp(gdf, lat, long):
    """
    Retrieves the index of the nearest geometry in a GeoDataFrame to a given point.

    The function cleans the latitude and longitude (if they are strings), converts them to floats,
    and adjusts their magnitude if necessary. A Shapely Point is created, and a new column 'distance'
    is added to the GeoDataFrame containing the distances between each geometry and the point.
    The index of the geometry with the minimum distance is then returned.

    Args:
        gdf (geopandas.GeoDataFrame): The GeoDataFrame containing geometries.
        lat (str or float): The latitude of the point.
        long (str or float): The longitude of the point.

    Returns:
        int: The index of the nearest geometry. Returns -1 if coordinate conversion fails.
    """
    if isinstance(lat, str):
        lat.replace(",", ".")
        lat.replace("'", "")
        lat.replace('"', "")
        try:
            lat = float(lat)
        except ValueError:
            return -1
    if isinstance(long, str):
        long.replace(",", ".")
        long.replace("'", "")
        long.replace('"', "")
        try:
            long = float(long)
        except ValueError:
            return -1
    # Adjust if latitude and longitude are of the wrong magnitude
    if lat > 10000:
        lat = lat / 1000
    if long > 10000:
        long = long / 1000
    point = Point(long, lat)
    # Calculate distance from each geometry to the point and store it in a new column
    gdf['distance'] = gdf.geometry.distance(point)
    index = gdf['distance'].idxmin()
    return index


def add_column_data_to_shp(gdf, attrs):
    """
    Adds new columns to a GeoDataFrame with an initial value of 0.

    For each attribute name provided in the list, a new column is created in the GeoDataFrame
    and all its values are initialized to 0.

    Args:
        gdf (geopandas.GeoDataFrame): The GeoDataFrame to update.
        attrs (list of str): A list of attribute names to add as columns.

    Returns:
        geopandas.GeoDataFrame: The updated GeoDataFrame with the new columns.
    """
    for att in attrs:
        gdf[att] = 0
    return gdf


def check_shp_in_shp_and_merge_data(gdf_container, gdf_contained, filed_name):
    """
    Merges attribute data from a container GeoDataFrame into a contained GeoDataFrame based on geometry intersection.

    For each geometry in the contained GeoDataFrame, the function determines if it intersects any geometry
    in the container GeoDataFrame. If an intersection is found, the value from the specified column in the first
    intersecting container geometry is assigned to the contained geometry. If no intersection is found, None is assigned.

    Args:
        gdf_container (geopandas.GeoDataFrame): The GeoDataFrame providing reference geometries and attribute data.
        gdf_contained (geopandas.GeoDataFrame): The GeoDataFrame to be updated with merged attribute data.
        filed_name (str): The column name whose value will be merged from the container to the contained GeoDataFrame.

    Returns:
        geopandas.GeoDataFrame: The contained GeoDataFrame updated with the merged attribute data.
    """
    gdf_contained[filed_name] = gdf_contained.geometry.apply(
        lambda x: gdf_container[gdf_container.intersects(x)][filed_name].iloc[0]
        if not gdf_container[gdf_container.intersects(x)].empty else None
    )
    return gdf_contained
