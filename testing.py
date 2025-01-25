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
import ollama
from functions import process_character_data  
from functions import process_ethnicity_data 
from functions import generate_characters_no_author



generate_characters_no_author(iterations=1)