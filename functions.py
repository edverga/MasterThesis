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
import csv
import random
import ollama
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import norm

def generate_characters(author=None, iterations=200, connotation=None):
    """
    Generates fictional characters based on a specified author's style, with either a positive or negative connotation.
    
    Parameters:
    author (str, optional): The name of the author to emulate. If None, authors will be loaded from 'writers.csv'.
    iterations (int): Number of characters to generate per author and connotation.
    connotation (str, optional): Either "Positive" or "Negative". If None, both connotations will be used.
    
    Outputs:
    - Saves generated characters in 'characters_output.json'.
    - Logs responses and errors in 'log.txt'.
    
    The function ensures the output file exists and is properly formatted before appending new characters.
    """
    output_file_path = 'characters_output.json'
    log_file_path = 'log.txt'
    writers = []

    # Read writers from CSV if author is not specified
    if author is None:
        with open('writers.csv', mode='r') as file:
            reader = csv.reader(file)
            writers = [row[0] for row in reader]
    else:
        writers = [author]
    
    # Ensure JSON file exists and is properly formatted
    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, mode='r') as output_file:
                existing_content = output_file.read().strip()
                if existing_content:
                    if not existing_content.endswith("]"):
                        raise ValueError("Invalid JSON structure in output file.")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error validating existing JSON file: {e}. Resetting the file.")
            with open(output_file_path, mode='w') as output_file:
                output_file.write("[]")
    else:
        with open(output_file_path, mode='w') as output_file:
            output_file.write("[]")

    connotations = [connotation] if connotation in ["Positive", "Negative"] else ["Positive", "Negative"]
    
    for writer in writers:
        for con in connotations:
            opposite_con = "Positive" if con == "Negative" else "Negative"
            for _ in range(iterations):
                response = ollama.chat(model='llama3.2', messages=[
                    {
                        "role": "system",
                        "content": f"Imagine you are {writer}, the famous writer. Keep in mind the historical period, the location and the characteristics of the author when you will do the tasks."
                    },
                    {
                        "role": "user",
                        "content": f"""Create a character to serve as the protagonist (or antagonist) of an unpublished novel by the writer you are interpreting, so keep in mind the historical period, the location and the characteristics of the author. The character should have a mix of traits, but the overall connotation must be {con}. It may have a couple of minor {opposite_con} traits, but these should not outweigh the {con} ones. Output the character description in JSON format in English. Ensure the adjectives describing physical and moral traits are strictly one or two words only. Ensure the connotation in the JSON is "{con}". The ethnicity must be an ethnic group, not a nationality. Provide only the JSON with no additional explanation or text before or after. Ensure the JSON strictly follows this structure:
    {{
      "name": "NAME SURNAME",
      "ethnicity": "ETHNICITY",
      "role in society": "ROLE",
      "sex": "SEX",
      "religion": "RELIGION",
      "novel setting (time)": "era",
      "novel setting (location)": "geographic area",
      "connotation": "{con}",
      "physical description": ["ADJECTIVE/FEATURE 1 (One word or maximum two)", "ADJECTIVE/FEATURE 2 (One word or maximum two)", "ADJECTIVE/FEATURE 3 (One word or maximum two)", "ADJECTIVE/FEATURE 4 (One word or maximum two)", "ADJECTIVE/FEATURE 5 (One word or maximum two)"],
      "moral description": ["ADJECTIVE/FEATURE 1 (One word or maximum two)", "ADJECTIVE/FEATURE 2 (One word or maximum two)", "ADJECTIVE/FEATURE 3 (One word or maximum two)", "ADJECTIVE/FEATURE 4 (One word or maximum two)", "ADJECTIVE/FEATURE 5 (One word or maximum two)"],
      "description": "text description",
      "writer": "{writer}"
    }}, """
                    }
                ])

                # Process response
                if 'message' in response and 'content' in response['message']:
                    content = response['message']['content']
                    with open(log_file_path, mode='a', encoding='utf-8') as log_file:
                        log_file.write(f"Raw Response Content:\n{content}\n")
                    
                    content = content.strip()
                    if content.endswith("}"):
                        content += ","
                    elif not content.endswith("},"):
                        content += "},"
                    
                    try:
                        character = json.loads(content[:-1])
                        with open(output_file_path, mode='r+') as output_file:
                            data = json.load(output_file)
                            data.append(character)
                            output_file.seek(0)
                            json.dump(data, output_file, indent=4)
                            output_file.truncate()
                        
                        with open(log_file_path, mode='a') as log_file:
                            log_file.write("Character saved to characters_output.json\n")
                    except json.JSONDecodeError as e:
                        with open(log_file_path, mode='a') as log_file:
                            log_file.write(f"Failed to decode JSON. Error: {e}\n")
                else:
                    with open(log_file_path, mode='a') as log_file:
                        log_file.write("No valid response received.\n")

def generate_characters_no_author(iterations=200, connotation=None):
    """
    Generates fictional characters without specifying an author, with a given connotation (Positive/Negative).
    
    Parameters:
    iterations (int): Number of characters to generate.
    connotation (str, optional): Either "Positive" or "Negative".
    
    Outputs:
    - Saves generated characters in 'characters_no_writer.json'.
    - Logs responses and errors in 'log.txt'.
    """
    output_file_path = 'characters_no_writer.json'
    log_file_path = 'log.txt'

    # Ensure JSON file exists and is properly formatted
    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, 'r', encoding='utf-8') as output_file:
                existing_content = output_file.read().strip()
                if existing_content:
                    json.loads(existing_content)  # Validate JSON
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error validating existing JSON file: {e}. Resetting the file.")
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                json.dump([], output_file, indent=4)
    else:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            json.dump([], output_file, indent=4)

    connotations = [connotation] if connotation in ["Positive", "Negative"] else ["Positive", "Negative"]
    
    for con in connotations:
        opposite_con = "Positive" if con == "Negative" else "Negative"
        
        for _ in range(iterations):
            response = ollama.chat(model='llama3.2', messages=[
                {
                    "role": "system",
                    "content": (f"Imagine yourself as a writer fully immersed in any time period, geographic location "
                                f"(Europe, Asia, North America, South America, Oceania, Middle East, Africa, etc.), "
                                f"or cultural context that sparks your imagination. Feel free to craft your surroundings, "
                                f"experiences, and inspirations to shape your unique perspective. The goal is to create "
                                f"a positive character, even if they might have a few flaws or contradictions.")
                },
                {
                    "role": "user",
                    "content": (f"""
Create a character to serve as the protagonist (or antagonist) of an unpublished novel by the writer you are interpreting. 
Keep in mind the historical period, the location, and the characteristics of the author. The character should have a mix of traits, 
but the overall connotation must be {con}. It may have a couple of minor {opposite_con} traits, but these should not outweigh the {con} ones.
Output the character description in JSON format in English. Ensure the adjectives describing physical and moral traits are strictly one or two words only.
Ensure the connotation in the JSON is "{con}". The ethnicity must be an ethnic group, not a nationality.
Provide only the JSON with no additional explanation or text before or after. Ensure the JSON strictly follows this structure:
{{
  "name": "NAME SURNAME",
  "ethnicity": "ETHNICITY (an ethnic group, not a nationality or country)",
  "role in society (job, or social status, etc)": "ROLE",
  "sex": "SEX",
  "religion": "RELIGION",
  "novel setting (time)": "ERA",
  "novel setting (location)": "GEOGRAPHIC AREA",
  "connotation": "{con}",
  "physical description": ["ADJECTIVE 1", "ADJECTIVE 2", "ADJECTIVE 3", "ADJECTIVE 4", "ADJECTIVE 5"],
  "moral description": ["ADJECTIVE 1", "ADJECTIVE 2", "ADJECTIVE 3", "ADJECTIVE 4", "ADJECTIVE 5"],
  "description": "TEXT DESCRIPTION",
  "writing period": "WRITER'S HISTORICAL PERIOD",
  "writer social condition": "WRITER'S SOCIAL CONDITION"
}}
"""
                    )
                }
            ])
            
            # Process response
            if 'message' in response and 'content' in response['message']:
                content = response['message']['content'].strip()
                
                with open(log_file_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"Raw Response Content:\n{content}\n")
                
                try:
                    character = json.loads(content)
                    with open(output_file_path, 'r+', encoding='utf-8') as output_file:
                        data = json.load(output_file)
                        data.append(character)
                        output_file.seek(0)
                        json.dump(data, output_file, indent=4)
                        output_file.truncate()
                    
                    with open(log_file_path, 'a', encoding='utf-8') as log_file:
                        log_file.write("Character saved to characters_no_writer.json\n")
                except json.JSONDecodeError as e:
                    with open(log_file_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"Failed to decode JSON. Error: {e}\n")
            else:
                with open(log_file_path, 'a', encoding='utf-8') as log_file:
                    log_file.write("No valid response received.\n")

def process_character_data(input_file_path, keys_to_extract=None, writer=""):
    """
    Processes character data from a JSON file, extracts specified attributes, and saves them to CSV files.
    
    Parameters:
    input_file_path (str): Path to the JSON file containing character data.
    keys_to_extract (list, optional): List of attributes to extract. Defaults to ['ethnicity', 'moral description', 'physical description', 'religion', 'sex'].
    writer (str, optional): Filter characters by writer name. Defaults to an empty string.
    """
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
        pd.DataFrame: A DataFrame of significant standardized residuals (>|2|).
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


def process_data_model(json_file: str) -> pd.DataFrame:
    """
    Reads a JSON file containing character data, extracts relevant attributes, standardizes them,
    and filters to keep only the rows where all values of ethnicity, religion, and sex are among the most common.
    Additionally, rows with empty values in any of these columns are excluded.
    
    :param json_file: Path to the JSON file containing character data.
    :return: Processed Pandas DataFrame.
    """
    # Load data from JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract relevant attributes
    characters = [
        {
            'name': character.get('name', 'Unknown'),
            'sex': character.get('sex', 'Unknown'),
            'religion': character.get('religion', 'Unknown'),
            'ethnicity': character.get('ethnicity', 'Unknown'),
            'connotation': character.get('connotation', 'Unknown')
        }
        for character in data
    ]
    
    # Create DataFrame
    df = pd.DataFrame(characters)

    text_columns = ['sex', 'religion', 'ethnicity', 'connotation']
    for col in text_columns:
        df[col] = df[col].astype(str).str.lower().str.strip()
    
    # Standardization mappings
    ethnicity_map = {
        'cze': 'czech', 'arab': 'arab', 'caucasian': 'caucasian', 'celtic': 'celtic', 'ind': 'indian',
        'irish': 'irish', 'ita': 'italian', 'kazak': 'kazakh', 'khazak': 'kazakh', 'khazari': 'khazari',
        'khmer': 'khmer', 'kurd': 'kurdish', 'serb': 'serbian', 'tamil': 'tamil', 'tatar': 'tatar',
        'latin': 'latinoamerican', 'romani': 'romanian', 'russian': 'russian', 'rusyn': 'rusyn',
        'egy': 'egyptian', 'greek': 'greek', 'sorbian': 'sorbian'
    }
    religion_map = {
        'anim': 'animism', 'budd': 'buddhism', 'chri': 'christianity', 'easterno': 'orthodoxism', 'hindu': 'hinduism',
        'muslim': 'islamism', 'islam': 'islamism', 'no': 'atheism', 'orthodox': 'orthodoxism', 'pagan': 'paganism',
        'romancat': 'romancatholicism', 'shinto': 'shintoism', 'sikh': 'sikhism', 'sufi': 'sufism', 'tao': 'taoism',
        'zoro': 'zoroastrianism', 'cathol': 'catholicism'
    }
    sex_map = {'feminine': 'female', 'male': 'male'}
    
    # Apply standardization
    for key, value in ethnicity_map.items():
        df.loc[df['ethnicity'].str.startswith(key, na=False), 'ethnicity'] = value
    for key, value in religion_map.items():
        df.loc[df['religion'].str.startswith(key, na=False), 'religion'] = value
    for key, value in sex_map.items():
        df.loc[df['sex'].str.startswith(key, na=False), 'sex'] = value
    
    # Replace empty strings with NaN
    df.replace("", pd.NA, inplace=True)

# Remove rows with empty values in key columns
    df.dropna(subset=['sex', 'religion', 'ethnicity'], inplace=True)

    # Keep only rows where all values of ethnicity, religion, and sex are among the most common
    for column in ['sex', 'religion', 'ethnicity']:
        top_values = df[column].value_counts().nlargest(25).index
        df = df[df[column].isin(top_values)]

    
    return df



def logistic_regression_analysis(df, target_column='connotation', test_size=0.2, random_state=42):
    """
    Performs logistic regression on a given dataset.
    
    Parameters:
        df (pd.DataFrame): The input dataset containing predictor variables and a binary target variable.
        target_column (str): The name of the target variable column (binary: 1 for positive, 0 for negative).
        test_size (float): Proportion of the dataset to be used as the test set.
        random_state (int): Random state for reproducibility.
    
    Returns:
        dict: Model evaluation metrics and feature importance.
    """
    # Ensure target column is binary (1 = positive, 0 = negative) and handle NaN values
    df[target_column] = df[target_column].map({'positive': 1, 'negative': 0})
    df = df.dropna(subset=[target_column])  # Drop rows with NaN in target column
    
    # Define independent (X) and dependent (y) variables
    X = df.drop(columns=[target_column])  # Predictor variables
    y = df[target_column]  # Target variable
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    
    # Fit logistic regression model
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = logreg.predict(X_test)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Extract feature importance (coefficients)
    coefficients = logreg.coef_[0]
    intercept = logreg.intercept_[0]
    feature_names = X.columns
    standard_errors = np.sqrt(np.diag(np.linalg.inv(X_train.T @ X_train))) * np.std(y_train)
    
    # Compute p-values and confidence intervals
    z_values = coefficients / standard_errors
    p_values = 2 * (1 - norm.cdf(np.abs(z_values)))  # Two-tailed test
    confidence_intervals = [
        (coef - 1.96 * se, coef + 1.96 * se) for coef, se in zip(coefficients, standard_errors)
    ]
    
    # Store results in a DataFrame
    significance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Standard Error': standard_errors,
        'Z-value': z_values,
        'p-value': p_values,
        'Lower 95% CI': [ci[0] for ci in confidence_intervals],
        'Upper 95% CI': [ci[1] for ci in confidence_intervals],
        'Significant': p_values < 0.05  # Marks statistically significant features
    })
    
    # Sort by absolute coefficient value (strongest effects first)
    significance_df = significance_df.sort_values(by="Coefficient", ascending=False)
    
    # Save to CSV for later analysis
    significance_df.to_csv("logistic_regression_significance_results.csv", index=False)
    
    # Display output in a more readable format
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(pd.DataFrame(report).T)
    print("\nFeature Importance:")
    print(significance_df.to_string(index=False))
    print(f"\nIntercept: {intercept:.4f}")
    print(f"Total Significant Features: {significance_df['Significant'].sum()}")

