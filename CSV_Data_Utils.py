import re
from geopy.geocoders import Nominatim

# Global variables for handling coordinates and address-related data
latitude = "LATITUDINE"
longitude = "LONGITUDINE"

address_data_pharmacy = ["INDIRIZZO", "CAP", "DESCRIZIONECOMUNE"]
name_data_pharmacy = "DESCRIZIONEFARMACIA"

address_data_parapharmacy = ["INDIRIZZO", "CAP", "DESCRIZIONECOMUNE"]
name_data_parapharmacy = "DENOMINAZIONESITOLOGISTICO"

address_data_hospital = ["Indirizzo", "Comune"]
name_data_hospital = "Denominazione struttura"

address_substring_to_remove = [
    'Viale', 'viale', 'VIALE', 'V.le', 'V.LE', 'Vial', 'vial', 'VIAL', 'via', 'Via', 'VIA',
    'corso', 'Corso', 'CORSO', 'C.so', 'piazza', 'Piazza', 'PIAZZA', 'P.zza', 'Piazzetta', 'Piazzale',
    'SNC', 'snc', 'Snc', 'Strada', 'STRADA', 'S.da', 'Località', 'località', 'LOCALITA', 'LOCALIT¿',
    'Largo', 'LARGO', 'largo', 'L.go', 'L.GO', 'Contrada', 'contrada', 'CONTRADA', 'C.da', 'C/da',
    'Vianuova', 'vianuova', 'VIANUOVA', 'Viacomunale', 'comunale', 'COMUNALE', 'Comunale', 'Nazionale',
    'NAZIONALE', 'nazionale', 'Vianazionale', 'Litoranea', 'LITORANEA', 'litoranea', 's.s.', 'Regionale',
    'regionale', 'REGIONALE', 'provinciale', 'Provinciale', 'PROVINCIALE', 'Superstrada', 'Stradaesterna',
    'REGIONE', 'regione', 'Regione'
]


def clear_duplicated_row(df, name_attr):
    """
    Removes duplicate rows from the DataFrame based on the specified column value.

    This method converts the values in the specified column to lowercase to ensure a
    case-insensitive comparison and then removes duplicates, keeping the first occurrence.

    Args:
        df (pandas.DataFrame): The DataFrame to be processed.
        name_attr (str): The name of the column to check for duplicates.

    Returns:
        pandas.DataFrame: The updated DataFrame with duplicate rows removed.
    """
    df[name_attr] = df[name_attr].str.lower()
    return df.drop_duplicates(subset=name_attr, keep='first')


def check_and_update_coordinates_parapharm_pharm(df, string_lat, string_long, address_data, name_data):
    """
    Checks and updates the coordinates (latitude and longitude) for pharmacies and parapharmacies.

    For each row in the DataFrame, the function checks if the coordinates (specified by
    'string_lat' and 'string_long') are missing or invalid (containing "-", "0", "0,0", etc.).
    If the coordinates are incorrect, the function attempts to retrieve the location using various
    approaches:
      - It first uses the full address (address + postal code + city). If the postal code has 4 digits,
        a zero is prepended.
      - If that fails, it attempts geolocation using the establishment's name.
      - If that also fails, it cleans the address by removing numbers, commas, slashes, special characters,
        and irrelevant substrings.
      - If this attempt also fails, it falls back to using city, postal code and province or just city and postal code.
      - If all methods fail, the coordinates are set to "-" and an error message is printed.

    The geolocator used is Nominatim from the geopy library.

    Args:
        df (pandas.DataFrame): DataFrame containing the data for the establishments.
        string_lat (str): The name of the column for latitude.
        string_long (str): The name of the column for longitude.
        address_data (list): A list of column names containing the address details (e.g., [address, postal code, city, ...]).
        name_data (str): The name of the column containing the establishment's name (pharmacy or parapharmacy).

    Returns:
        pandas.DataFrame: The updated DataFrame with the correct coordinates.
    """
    loc = Nominatim(user_agent="Geopy Library")

    for idx, row in df.iterrows():
        print(idx)
        if row[string_lat] == "-" or row[string_lat] == "0" or row[string_lat] == "0,0" or "-" in row[string_lat] or "-" in row[string_long]:
            cap_info = str(row[address_data[1]])
            if len(cap_info) == 4:
                cap_info = "0" + cap_info
            curr_addr = row[address_data[0]] + " " + str(cap_info) + " " + row[address_data[2]]
            curr_name = row[name_data]
            try:
                # Attempt geolocation using the full address
                getLoc = loc.geocode(curr_addr)
                df.at[idx, string_lat] = getLoc.latitude
                df.at[idx, string_long] = getLoc.longitude
            except:
                try:
                    # Attempt geolocation using the establishment's name
                    getLoc = loc.geocode(curr_name)
                    df.at[idx, string_lat] = getLoc.latitude
                    df.at[idx, string_long] = getLoc.longitude
                except:
                    try:
                        # Clean the address and try geolocation again
                        temp_addr = row[address_data[0]]
                        temp_addr = re.sub(r'\d+', '', temp_addr)  # remove numbers
                        temp_addr = re.sub(r',', '', temp_addr)      # remove commas
                        temp_addr = re.sub(r'/', '', temp_addr)      # remove slashes
                        temp_addr = re.sub(r'`', '', temp_addr)       # remove backticks
                        temp_addr = re.sub(r'¿', '', temp_addr)       # remove special characters
                        temp_addr = temp_addr.replace('o\'', 'ò')
                        temp_addr = temp_addr.replace('a\'', 'à')
                        for s in address_substring_to_remove:  # remove street type substrings
                            temp_addr = temp_addr.replace(s, '')
                        while temp_addr.find('.') != -1:  # remove dots and the previous character
                            dot_index = temp_addr.find('.')
                            if dot_index != 0:
                                temp_addr = temp_addr[:dot_index - 1] + temp_addr[dot_index + 1:]
                            else:
                                temp_addr = temp_addr[1:]
                        curr_addr = temp_addr + " " + row[address_data[2]] + " " + cap_info

                        getLoc = loc.geocode(curr_addr)
                        df.at[idx, string_lat] = getLoc.latitude
                        df.at[idx, string_long] = getLoc.longitude
                    except:
                        try:
                            # Fallback: geolocate using city, postal code and province
                            curr_addr = row[address_data[2]] + " " + cap_info + " " + row[address_data[3]]
                            getLoc = loc.geocode(curr_addr)
                            df.at[idx, string_lat] = getLoc.latitude
                            df.at[idx, string_long] = getLoc.longitude
                        except:
                            try:
                                # Fallback: geolocate using only city and postal code
                                curr_addr = row[address_data[2]] + " " + cap_info
                                getLoc = loc.geocode(curr_addr)
                                df.at[idx, string_lat] = getLoc.latitude
                                df.at[idx, string_long] = getLoc.longitude
                            except:
                                df.at[idx, string_lat] = "-"
                                df.at[idx, string_long] = "-"
                                print("ERROR for " + curr_name + " - Addr:" + curr_addr + " - Lat:" + row[string_lat] + " - Long:" + row[string_long])
        else:
            if isinstance(df.at[idx, string_lat], str):
                if "'" in df.at[idx, string_lat]:
                    df.at[idx, string_lat] = df.at[idx, string_lat].replace("'", "")
                if "," in df.at[idx, string_lat]:
                    df.at[idx, string_lat] = df.at[idx, string_lat].replace(",", ".")
                    df.at[idx, string_lat] = float(df.at[idx, string_lat])

            if isinstance(df.at[idx, string_long], str):
                if "'" in df.at[idx, string_long]:
                    df.at[idx, string_long] = df.at[idx, string_long].replace("'", "")
                if "," in df.at[idx, string_long]:
                    df.at[idx, string_long] = df.at[idx, string_long].replace(",", ".")
                    df.at[idx, string_long] = float(df.at[idx, string_long])
    return df


def get_coordinates_hospital(df, string_lat, string_long, address_data, name_data):
    """
    Retrieves the coordinates (latitude and longitude) for hospital facilities.

    The function attempts to geolocate a hospital using:
      - The full address (combining address and city).
      - The facility's name if the address does not yield a valid geolocation.
      - A modified version of the address (with numbers, commas, slashes, and other anomalous characters removed).
      - As a further fallback, it attempts geolocation based solely on the city.

    Args:
        df (pandas.DataFrame): DataFrame containing hospital data.
        string_lat (str): The name of the column for latitude.
        string_long (str): The name of the column for longitude.
        address_data (list): A list of column names related to the address (e.g., [Address, City]).
        name_data (str): The name of the column containing the hospital's name.

    Returns:
        pandas.DataFrame: The updated DataFrame with the obtained coordinates.
    """
    df[string_lat] = ''
    df[string_long] = ''

    loc = Nominatim(user_agent="Geopy Library")

    for idx, row in df.iterrows():
        curr_addr = row[address_data[0]] + " " + row[address_data[1]]
        curr_name = row[name_data]
        try:
            getLoc = loc.geocode(curr_addr)
            df.at[idx, string_lat] = getLoc.latitude
            df.at[idx, string_long] = getLoc.longitude
        except:
            try:
                getLoc = loc.geocode(curr_name)
                df.at[idx, string_lat] = getLoc.latitude
                df.at[idx, string_long] = getLoc.longitude
            except:
                try:
                    temp_addr = row[address_data[0]]
                    temp_addr = re.sub(r'\d+', '', temp_addr)  # remove numbers
                    temp_addr = re.sub(r',', '', temp_addr)      # remove commas
                    temp_addr = re.sub(r'/', '', temp_addr)      # remove slashes
                    temp_addr = re.sub(r'`', '', temp_addr)       # remove backticks
                    temp_addr = re.sub(r'¿', '', temp_addr)       # remove special characters
                    temp_addr = temp_addr.replace('o\'', 'ò')
                    temp_addr = temp_addr.replace('a\'', 'à')
                    for s in address_substring_to_remove:
                        temp_addr = temp_addr.replace(s, '')
                    while temp_addr.find('.') != -1:
                        dot_index = temp_addr.find('.')
                        if dot_index != 0:
                            temp_addr = temp_addr[:dot_index - 1] + temp_addr[dot_index + 1:]
                        else:
                            temp_addr = temp_addr[1:]
                    curr_addr = temp_addr

                    getLoc = loc.geocode(curr_addr)
                    df.at[idx, string_lat] = getLoc.latitude
                    df.at[idx, string_long] = getLoc.longitude
                except:
                    try:
                        # Fallback: geolocation using only the city
                        curr_addr = row[address_data[1]]
                        getLoc = loc.geocode(curr_addr)
                        df.at[idx, string_lat] = getLoc.latitude
                        df.at[idx, string_long] = getLoc.longitude
                    except:
                        df.at[idx, string_lat] = "-"
                        df.at[idx, string_long] = "-"
                        print("ERROR for " + curr_name + " - Addr:" + curr_addr + " - Lat:" + row[string_lat] + " - Long:" + row[string_long])
    return df


def drop_location_with_no_coordinates(df):
    """
    Removes the rows from the DataFrame that do not have valid coordinates.

    Rows where the latitude or longitude columns contain the value "-"
    are removed from the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing locations with potential missing coordinates.

    Returns:
        pandas.DataFrame: A filtered DataFrame containing only rows with valid coordinates.
    """
    df = df[df[latitude] != "-"]
    df = df[df[longitude] != "-"]
    return df


def extract_location_with_no_coordinates(df):
    """
    Extracts the rows from the DataFrame that do not have valid coordinates.

    Filters the DataFrame to select only those rows where the latitude and longitude
    columns contain the value "-" (indicating missing or invalid coordinates).

    Args:
        df (pandas.DataFrame): DataFrame containing locations with invalid coordinates.

    Returns:
        pandas.DataFrame: A DataFrame containing only the rows without valid coordinates.
    """
    df = df[df[latitude] == "-"]
    df = df[df[longitude] == "-"]
    return df
