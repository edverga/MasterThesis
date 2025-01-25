# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy>=2.2.2",
#     "pandas>=2.2.3",
#     "scipy>=1.15.1",
# ]
# ///

import json
import pandas as pd
import os
import scipy.stats as stats
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import numpy as np
from scipy.special import rel_entr

def process_character_data(input_file_path, keys_to_extract=None, writer=""):
    if keys_to_extract is None:
        keys_to_extract = ['ethnicity', 'moral description', 'physical description', 'religion', 'sex']
    
    def format_text(text):
        return text.lower().replace(" ", "") if isinstance(text, str) else "unknown"
    
    try:
        # Load the JSON data
        with open(input_file_path, mode='r', encoding='utf-8') as input_file:
            data = json.load(input_file)
        
        # Ensure the data is a list of characters
        if not isinstance(data, list):
            print("Error: JSON data is not a list of characters.")
            return
        
        # Create the 'csv' folder if it doesn't exist
        csv_folder = "csv"
        os.makedirs(csv_folder, exist_ok=True)
        
        # Apply author filter if specified
        if writer:
            data = [char for char in data if char.get("writer", "").lower() == writer.lower()]
            if not data:
                print(f"No characters found for writer: {writer}")
                return
            writer_surname = writer.split()[-1].lower()
        else:
            writer_surname = ""
        
        for key in keys_to_extract:
            rows = []
            
            # Iterate through each character
            for character in data:
                connotation = format_text(character.get("connotation", "unknown"))
                value = character.get(key, "unknown")
                
                # If the value is a list, extract each item separately
                if isinstance(value, list):
                    for item in value:
                        rows.append({key.replace(" ", "").lower(): format_text(item), "connotation": connotation})
                else:
                    rows.append({key.replace(" ", "").lower(): format_text(value), "connotation": connotation})
            
            # Convert to a pandas DataFrame
            df = pd.DataFrame(rows)
            
            # Define the output file path inside the 'csv' folder
            output_file_name = f"{key.replace(' ', '').lower()}_data"
            if writer_surname:
                output_file_name += f"_{writer_surname}"
            output_file_name += ".csv"
            
            output_file_path = os.path.join(csv_folder, output_file_name)  # Save inside 'csv' folder
            
            df.to_csv(output_file_path, index=False, encoding='utf-8')
            print(f"{key} data saved to {output_file_path}")
            
            # Read the CSV file
            df = pd.read_csv(output_file_path, header=0)
            
            # Normalize column names for processing (assuming there are only two columns: value and connotation)
            value_col = df.columns[0]
            connotation_col = df.columns[1]
            
            # Convert all text to lowercase and remove spaces
            df[value_col] = df[value_col].str.lower().str.replace(r'\s+', '', regex=True)
            df[connotation_col] = df[connotation_col].str.lower().str.replace(r'\s+', '', regex=True)
            
            # Count the occurrences of each unique row
            counts = df.groupby([value_col, connotation_col]).size()
            
            # Reset index to convert grouped data into a DataFrame and name the count column
            result = counts.reset_index(name='count')
            
            # Sort the DataFrame alphabetically by value_col and then by connotation and count
            adf = result.sort_values(by=[value_col, 'connotation', 'count'], ascending=[True, True, False])
            
            # Define processed CSV file path inside the 'csv' folder
            processed_output_file_name = f"processed_{output_file_name}"
            processed_output_file_path = os.path.join(csv_folder, processed_output_file_name)
            
            # Save the processed DataFrame back to CSV
            adf.to_csv(processed_output_file_path, index=False, encoding='utf-8')
            print(f"Processed {key} data saved to {processed_output_file_path}")
    
    except FileNotFoundError:
        print(f"Error: File {input_file_path} not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file_path}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def process_ethnicity_data(csv_file):
    # Load the dataset
    ethnicity = pd.read_csv(csv_file)
    
    # Standardizing ethnicity values
    standardization_map = {
        'cze': 'czech', 'arab': 'arab', 'caucasian': 'caucasian', 'celtic': 'celtic', 'ind': 'indian',
        'irish': 'irish', 'ita': 'italian', 'kazak': 'kazakh', 'khazak': 'kazakh', 'khazari': 'khazari',
        'khmer': 'khmer', 'kurd': 'kurdish', 'serb': 'serbian', 'tamil': 'tamil', 'tatar': 'tatar',
        'latin': 'latinoamerican', 'romani': 'romanian', 'russian': 'russian', 'rusyn': 'rusyn',
        'egy': 'egyptian', 'greek': 'greek', 'sorbian': 'sorbian'
    }
    
    for key, value in standardization_map.items():
        ethnicity.loc[ethnicity['ethnicity'].str.startswith(key, na=False), 'ethnicity'] = value
    
    # Perform grouping and sum the 'count' column
    ethnicity = ethnicity.groupby(['ethnicity', 'connotation'], as_index=False)['count'].sum()
    
    # Compute total counts per connotation
    total_counts = ethnicity.groupby('connotation')['count'].sum().reset_index()
    total_counts.rename(columns={'count': 'total_connotation_count'}, inplace=True)
    
    # Merge total counts with main DataFrame
    ethnicity = ethnicity.merge(total_counts, on='connotation')
    
    # Calculate relative frequency
    ethnicity['relative_frequency'] = ethnicity['count'] / ethnicity['total_connotation_count']
    
    # Drop the total count column (optional)
    ethnicity.drop(columns=['total_connotation_count'], inplace=True)
    
    # Sort by connotation and relative frequency in descending order
    ethnicity = ethnicity.sort_values(by=['connotation', 'relative_frequency'], ascending=[True, False])
    
    # Filter for relative frequency >= 0.01
    ethnicity = ethnicity[ethnicity['relative_frequency'] >= 0.01]
    
    # Return processed dataframe
    return ethnicity

def chi_square_test(dataset, index_col=''):
    """
    Perform a Chi-Square test on the given ethnicity DataFrame.
    
    Parameters:
        dataset (pd.DataFrame): DataFrame with columns ['characteristic', 'connotation', 'count'].
        index_col (str): The column to use as the index for the contingency table.

    Returns:
        dict: A dictionary containing the Chi-Square statistic, p-value, degrees of freedom, and expected frequencies.
    """
    # Aggregate counts by ethnicity and connotation
    dataset = dataset.groupby([index_col, 'connotation'], as_index=False)['count'].sum()
    
    # Create a contingency table
    contingency_table = dataset.pivot(index=index_col, columns='connotation', values='count').fillna(0)
    
    # Perform the Chi-Square test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Return results as a dictionary
    return {
        "Chi-Square Statistic": round(chi2_stat, 4),
        "p-value": round(p_value, 4),
        "Degrees of Freedom": dof
    }

def compute_standardized_residuals(dataset, index_col=''):
    """
    Compute standardized residuals from a Chi-Square test.
    
    Parameters:
        dataset (pd.DataFrame): DataFrame with columns ['characteristic', 'connotation', 'count'].
        index_col (str): The column to use as the index for the contingency table.

    Returns:
        pd.DataFrame: A DataFrame of significant standardized residuals (>|2| or <|-2|).
    """
    # Aggregate counts by ethnicity and connotation
    dataset = dataset.groupby([index_col, 'connotation'], as_index=False)['count'].sum()
    
    # Create a contingency table
    contingency_table = dataset.pivot(index=index_col, columns='connotation', values='count').fillna(0)
    
    # Perform the Chi-Square test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Convert expected values into a DataFrame
    expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
    
    # Compute standardized residuals
    standardized_residuals = (contingency_table - expected_df) / (expected_df ** 0.5)
    
    # Identify significant residuals (values > 2 or < -2)
    significant_residuals = standardized_residuals[(standardized_residuals > 2) | (standardized_residuals < -2)]

    significant_residuals = significant_residuals.dropna(how='any')
    
    return significant_residuals

def compute_divergences(dataset, connotations, index_col=''):
    """
    Compute Jensen-Shannon Divergence (JSD) and Kullback-Leibler Divergence (KL) between connotation distributions.
    """
    connotation_distributions = {}
    for con in connotations:
        subset = dataset[dataset["connotation"] == con]
        distribution = subset.groupby(index_col)["relative_frequency"].sum()
        distribution = distribution / distribution.sum()
        connotation_distributions[con] = distribution
    jsd_results = {}
    kl_results = {}
    connotation_list = list(connotation_distributions.keys())
    for i in range(len(connotation_list)):
        for j in range(i + 1, len(connotation_list)):
            con1, con2 = connotation_list[i], connotation_list[j]
            all_values = set(connotation_distributions[con1].index).union(set(connotation_distributions[con2].index))
            p = np.array([connotation_distributions[con1].get(e, 0) for e in all_values])
            q = np.array([connotation_distributions[con2].get(e, 0) for e in all_values])
            jsd_results[f"JSD({con1} || {con2})"] = jensenshannon(p, q, base=2)
            epsilon = 1e-10
            p = np.where(p == 0, epsilon, p)
            q = np.where(q == 0, epsilon, q)
            kl_results[f"KL({con1} || {con2})"] = np.sum(rel_entr(p, q))
            kl_results[f"KL({con2} || {con1})"] = np.sum(rel_entr(q, p))
    return {"JSD": jsd_results, "KL": kl_results}

def process_religion_data(csv_file):
    # Load the dataset
    religion = pd.read_csv(csv_file)
    
    # Standardizing ethnicity values
    standardization_map = religion_mapping = {
    'anim': 'animism', 'budd': 'buddhism', 'chri': 'christianity', 'easterno': 'orthodoxism', 'hindu': 'hinduism', 'muslim': 'islamism', 'islam': 'islamism', 'no': 'atheism', 'orthodox': 'orthodoxism', 'pagan': 'paganism', 'romancat': 'romancatholicism', 'shinto': 'shintoism',
    'sikh': 'sikhism', 'sufi': 'sufism', 'tao': 'taoism', 'zoro': 'zoroastrianism', 'cathol': 'catholicism'
}
    
    for key, value in standardization_map.items():
        religion.loc[religion['religion'].str.startswith(key, na=False), 'religion'] = value
    
    # Perform grouping and sum the 'count' column
    religion = religion.groupby(['religion', 'connotation'], as_index=False)['count'].sum()
    
    # Compute total counts per connotation
    total_counts = religion.groupby('connotation')['count'].sum().reset_index()
    total_counts.rename(columns={'count': 'total_connotation_count'}, inplace=True)
    
    # Merge total counts with main DataFrame
    religion = religion.merge(total_counts, on='connotation')
    
    # Calculate relative frequency
    religion['relative_frequency'] = religion['count'] / religion['total_connotation_count']
    
    # Drop the total count column (optional)
    religion.drop(columns=['total_connotation_count'], inplace=True)
    
    # Sort by connotation and relative frequency in descending order
    religion = religion.sort_values(by=['connotation', 'relative_frequency'], ascending=[True, False])
    
    # Filter for relative frequency >= 0.01
    religion = religion[religion['relative_frequency'] >= 0.01]
    
    # Return processed dataframe
    return religion

def process_sex_data(csv_file):
    # Load the dataset
    sex = pd.read_csv(csv_file)
    
    # Standardizing ethnicity values
    standardization_map = {
        'feminine': 'female'
    }
    
    for key, value in standardization_map.items():
        sex.loc[sex['sex'].str.startswith(key, na=False), 'sex'] = value
    
    # Perform grouping and sum the 'count' column
    sex = sex.groupby(['sex', 'connotation'], as_index=False)['count'].sum()
    
    # Compute total counts per connotation
    total_counts = sex.groupby('connotation')['count'].sum().reset_index()
    total_counts.rename(columns={'count': 'total_connotation_count'}, inplace=True)
    
    # Merge total counts with main DataFrame
    sex = sex.merge(total_counts, on='connotation')
    
    # Calculate relative frequency
    sex['relative_frequency'] = sex['count'] / sex['total_connotation_count']
    
    # Drop the total count column (optional)
    sex.drop(columns=['total_connotation_count'], inplace=True)
    
    # Sort by connotation and relative frequency in descending order
    sex = sex.sort_values(by=['connotation', 'relative_frequency'], ascending=[True, False])
        
    # Return processed dataframe
    return sex

def process_phy_data(csv_file):
    # Load the dataset
    phy = pd.read_csv(csv_file)
        
    standardization_map = phy_mapping = {
    
    }
    for key, value in standardization_map.items():
        phy.loc[phy['physicaldescription'].str.startswith(key, na=False), 'physicaldescription'] = value
    
    # Perform grouping and sum the 'count' column
    phy = phy.groupby(['physicaldescription', 'connotation'], as_index=False)['count'].sum()
    
    # Compute total counts per connotation
    total_counts = phy.groupby('connotation')['count'].sum().reset_index()
    total_counts.rename(columns={'count': 'total_connotation_count'}, inplace=True)
    
    # Merge total counts with main DataFrame
    phy = phy.merge(total_counts, on='connotation')
    
    # Calculate relative frequency
    phy['relative_frequency'] = phy['count'] / phy['total_connotation_count']
    
    # Drop the total count column (optional)
    phy.drop(columns=['total_connotation_count'], inplace=True)
    
    # Sort by connotation and relative frequency in descending order
    phy = phy.sort_values(by=['connotation', 'relative_frequency'], ascending=[True, False])
    
    # Filter for relative frequency >= 0.01
    phy = phy[phy['relative_frequency'] >= 0.005]
    
    # Return processed dataframe
    return phy

def process_mor_data(csv_file):
    # Load the dataset
    mor = pd.read_csv(csv_file)
        
    standardization_map = mor_mapping = {
    
    }
    for key, value in standardization_map.items():
        mor.loc[mor['moraldescription'].str.startswith(key, na=False), 'moraldescription'] = value
    
    # Perform grouping and sum the 'count' column
    mor = mor.groupby(['moraldescription', 'connotation'], as_index=False)['count'].sum()
    
    # Compute total counts per connotation
    total_counts = mor.groupby('connotation')['count'].sum().reset_index()
    total_counts.rename(columns={'count': 'total_connotation_count'}, inplace=True)
    
    # Merge total counts with main DataFrame
    mor = mor.merge(total_counts, on='connotation')
    
    # Calculate relative frequency
    mor['relative_frequency'] = mor['count'] / mor['total_connotation_count']
    
    # Drop the total count column (optional)
    mor.drop(columns=['total_connotation_count'], inplace=True)
    
    # Sort by connotation and relative frequency in descending order
    mor = mor.sort_values(by=['connotation', 'relative_frequency'], ascending=[True, False])
    
    # Filter for relative frequency >= 0.01
    mor = mor[mor['relative_frequency'] >= 0.005]
    
    # Return processed dataframe
    return mor

def chi_square_test_authors(dataset, index_col=''):
    """
    Perform a Chi-Square test on the given ethnicity DataFrame.
    
    Parameters:
        dataset (pd.DataFrame): DataFrame with columns ['characteristic', 'author', 'count'].
        index_col (str): The column to use as the index for the contingency table.

    Returns:
        dict: A dictionary containing the Chi-Square statistic, p-value, degrees of freedom, and expected frequencies.
    """
    # Aggregate counts by ethnicity and connotation
    dataset = dataset.groupby([index_col, 'author'], as_index=False)['count'].sum()
    
    # Create a contingency table
    contingency_table = dataset.pivot(index=index_col, columns='author', values='count').fillna(0)
    
    # Perform the Chi-Square test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Return results as a dictionary
    return {
        "Chi-Square Statistic": round(chi2_stat, 4),
        "p-value": round(p_value, 4),
        "Degrees of Freedom": dof
    }

def compute_standardized_residuals_authors(dataset, index_col=''):
    """
    Compute standardized residuals from a Chi-Square test.
    
    Parameters:
        dataset (pd.DataFrame): DataFrame with columns ['characteristic', 'connotation', 'count'].
        index_col (str): The column to use as the index for the contingency table.

    Returns:
        pd.DataFrame: A DataFrame of significant standardized residuals (>|2| or <|-2|).
    """
    # Aggregate counts by ethnicity and connotation
    dataset = dataset.groupby([index_col, 'author'], as_index=False)['count'].sum()
    
    # Create a contingency table
    contingency_table = dataset.pivot(index=index_col, columns='author', values='count').fillna(0)
    
    # Perform the Chi-Square test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Convert expected values into a DataFrame
    expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
    
    # Compute standardized residuals
    standardized_residuals = (contingency_table - expected_df) / (expected_df ** 0.5)
    
    # Identify significant residuals (values > 2 or < -2)
    significant_residuals = standardized_residuals[(standardized_residuals > 2) | (standardized_residuals < -2)]

    significant_residuals = significant_residuals.dropna(how='any')
    
    return significant_residuals

def compute_divergences_authors(dataset, authors, index_col=''):
    """
    Compute Jensen-Shannon Divergence (JSD) and Kullback-Leibler Divergence (KL) between connotation distributions.
    """
    author_distributions = {}
    for con in authors:
        subset = dataset[dataset["author"] == con]
        distribution = subset.groupby(index_col)["relative_frequency"].sum()
        distribution = distribution / distribution.sum()
        author_distributions[con] = distribution
    jsd_results = {}
    kl_results = {}
    author_list = list(author_distributions.keys())
    for i in range(len(author_list)):
        for j in range(i + 1, len(author_list)):
            con1, con2 = author_list[i], author_list[j]
            all_values = set(author_distributions[con1].index).union(set(author_distributions[con2].index))
            p = np.array([author_distributions[con1].get(e, 0) for e in all_values])
            q = np.array([author_distributions[con2].get(e, 0) for e in all_values])
            jsd_results[f"JSD({con1} || {con2})"] = jensenshannon(p, q, base=2)
            epsilon = 1e-10
            p = np.where(p == 0, epsilon, p)
            q = np.where(q == 0, epsilon, q)
            kl_results[f"KL({con1} || {con2})"] = np.sum(rel_entr(p, q))
            kl_results[f"KL({con2} || {con1})"] = np.sum(rel_entr(q, p))
    return {"JSD": jsd_results, "KL": kl_results}


