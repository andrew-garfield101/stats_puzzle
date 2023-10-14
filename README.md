# Analysis using Lottery Drawings

## Overview

Exploring lottery draw patterns using machine learning and stats, based on a Socrata dataset.

## Tech

- Python
- Pandas for data manipulation
- Matplotlib for data visualization
- Scipy for chi-squared statistical tests

## Getting Started

Create a username / password and app token for Socrata dataset use. 
See https://data.ny.gov/login for more 

1. Clone the repository
2. Navigate to the project folder: `cd stats_puzzle`
3. Install the required packages: `pip install -r requirements.txt`
4. Run the `main.py` script to collect data: `python main.py`
5. Run the `chi_tests.py` script for chi-squared tests: `python chi_tests.py`

## Features

- **Data Collection**: Gathers historical draw data for lottery numbers.
- **Statistical Analysis**: Performs chi-squared tests on the observed vs expected frequencies of lottery numbers.
- **Data Visualization**: Plots observed vs expected frequencies for both white balls and Powerballs.



