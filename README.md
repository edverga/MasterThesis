# Master Thesis: Assessing Bias in Creative Tasks Using LLMs

This repository contains all scripts, functions, and data used in my Master's thesis, *Assessing Bias in Creative Tasks Using LLMs: An NST Approach*. The research analyzes bias in Large Language Models (LLMs) by generating characters with predefined attributes and evaluating the distribution of positive/negative adjectives across categorical variables such as gender, ethnicity, religion, and physical characteristics.

## Repository Structure

### Code
- **`functions.py`** – Contains the functions used to generate and analyze characters using LLaMa.
- **`thesis_draft.ipynb`** – The current working notebook for analysis and evaluation.
- **`testing.py`** – Can be used to test the functions in a separate environment.

### Data
- **`writers.csv`** – Stores the list of writers used in prompts.
- **`characters_no_writer.json`** – JSON file containing generated character data without writer attribution.
- **`characters_output.json`** – JSON file with the final generated character outputs.

