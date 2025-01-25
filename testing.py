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
from functions import process_character_data  
from functions import process_ethnicity_data 
process_character_data(r"C:\Users\edoar\Documents\Tesi\characters_no_writer.json", keys_to_extract=None, writer="")
df = process_ethnicity_data(r"C:\Users\edoar\Documents\Tesi\thesis\csv\processed_ethnicity_data.csv")
print(df)
