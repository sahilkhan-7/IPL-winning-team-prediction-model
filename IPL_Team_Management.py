# %%
import os
import csv

# Add the "Year" column to the dataset
def add_year_column(dataset, year):
    for entry in dataset:
        entry['Year'] = year

# Load and merge data from all files in a folder
def load_and_merge_data(folder_path):
    merged_data = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as csvfile:
            csvreader = csv.DictReader(csvfile)
            year = int(filename.split('_')[-1].split('.')[0])  # Extract year from filename
            for row in csvreader:
                row['Year'] = year
                merged_data.append(row)
    return merged_data

# Load and merge bowling statistics from all files
bowling_folder_path = 'IPL Player Stats/Bowling Stats'  # Adjust the folder path
merged_bowling_stats = load_and_merge_data(bowling_folder_path)

# Load and merge batting statistics from all files
batting_folder_path = 'IPL Player Stats/Batting Stats'  # Adjust the folder path
merged_batting_stats = load_and_merge_data(batting_folder_path)

# Print merged bowling statistics
print("Merged Bowling Stats:")
for entry in merged_bowling_stats[:5]:
    print(entry)

# Print merged batting statistics
print("\nMerged Batting Stats:")
for entry in merged_batting_stats[:5]:
    print(entry)

# %%
def describe_dataset(dataset):
    # Calculate dimensions
    num_rows = len(dataset)
    num_columns = len(dataset[0])

    # Print dataset dimensions
    print(f"Dataset Dimensions: {num_rows} rows x {num_columns} columns\n")

    # Print column information
    print("Column Name | Data Type | Non-Null Counts")
    print("-" * 40)
    for column_name in dataset[0].keys():
        data_type = type(dataset[0][column_name]).__name__
        non_null_count = sum(1 for entry in dataset if entry[column_name] != '')
        print(f"{column_name:12} | {data_type:10} | {non_null_count:15}")

# Call the describe function for merged batting and bowling stats
describe_dataset(merged_batting_stats)  # Replace with your merged dataset
describe_dataset(merged_bowling_stats)  # Replace with your merged dataset

# %%
def clean_dataset(dataset):
    cleaned_data = []
    for entry in dataset:
        # Remove rows with missing values
        if all(value != '' for value in entry.values()):
            # Clean HS column
            if 'HS' in entry:
                hs_value = entry['HS'].replace('*', '')
                entry['HS'] = int(hs_value) if hs_value.isdigit() else 0
            
            # Replace other columns with '-' or 'NA' with 0
            for column in entry:
                if entry[column] in ('-', 'NA', 'None'):
                    entry[column] = 0
            
            cleaned_data.append(entry)
    return cleaned_data

# Clean merged batting statistics
cleaned_batting_stats = clean_dataset(merged_batting_stats)  # Replace with your merged dataset

# Clean merged bowling statistics
cleaned_bowling_stats = clean_dataset(merged_bowling_stats)  # Replace with your merged dataset

# Print first few entries from cleaned datasets
print("Cleaned Batting Stats:")
for entry in cleaned_batting_stats[:5]:
    print(entry)

print("\nCleaned Bowling Stats:")
for entry in cleaned_bowling_stats[:5]:
    print(entry)

# %%
# Function to remove players who played only once
def remove_single_appearance_players(dataset):
    player_counts = {}  # Dictionary to store player appearance counts
    
    # Count player appearances
    for entry in dataset:
        player = entry['Player']
        if player in player_counts:
            player_counts[player] += 1
        else:
            player_counts[player] = 1
    
    # Filter out players with only one appearance
    filtered_dataset = [entry for entry in dataset if player_counts[entry['Player']] > 1]
    return filtered_dataset

# Remove single appearance players from batting and bowling stats datasets
cleaned_bowling_stats = remove_single_appearance_players(cleaned_bowling_stats)
cleaned_batting_stats = remove_single_appearance_players(cleaned_batting_stats)

# %%
def convert_to_numeric(dataset):
    for entry in dataset:
        for column_name in entry:
            value = entry[column_name]
            try:
                if '.' in value:
                    entry[column_name] = float(value)
                else:
                    entry[column_name] = int(value)
            except (ValueError, TypeError):
                pass

# Convert columns to numeric type for cleaned batting and bowling stats
convert_to_numeric(cleaned_batting_stats)
convert_to_numeric(cleaned_bowling_stats)

# Print first few entries from cleaned and converted datasets
print("Cleaned and Converted Batting Stats:")
for entry in cleaned_batting_stats[:5]:
    print(entry)

print("\nCleaned and Converted Bowling Stats:")
for entry in cleaned_bowling_stats[:5]:
    print(entry)

# %%
describe_dataset(cleaned_batting_stats)

# %%
describe_dataset(cleaned_bowling_stats)

# %%
from collections import defaultdict

# Function to get top N players based on a specific key
def get_top_players(dataset, key, n=40, reverse=True):
    return sorted(dataset, key=lambda x: x[key], reverse=reverse)[:n]

# Get top performing players year-wise for all metrics
def get_top_players_year_wise(dataset, metrics, n=50):
    top_players_year_wise = defaultdict(lambda: defaultdict(list))

    for player in dataset:
        year = player['Year']
        for metric in metrics:
            top_players_year_wise[year][metric].append(player)

    return top_players_year_wise

# List of metrics for batsmen and bowlers
batsman_metrics = ['SR', 'HS', 'Runs', 'Avg', '4s', '6s', '100']
bowler_metrics = ['Wkts', 'Ov', 'Econ', 'SR', '4w', '5w']

# Get top performing batsmen and bowlers year-wise for all metrics
top_batsmen_year_wise = get_top_players_year_wise(cleaned_batting_stats, batsman_metrics)
top_bowlers_year_wise = get_top_players_year_wise(cleaned_bowling_stats, bowler_metrics)

# Print top performing batsmen year-wise for all metrics
for year, metrics in top_batsmen_year_wise.items():
    print(f"Year: {year}")
    for metric, players in metrics.items():
        top_players = get_top_players(players, metric)
        print(f"{metric}:")
        for i, player in enumerate(top_players, start=1):
            print(f"{i}. {player['Player']} - {player[metric]:.2f}")
        print()

# Print top performing bowlers year-wise for all metrics
for year, metrics in top_bowlers_year_wise.items():
    print(f"Year: {year}")
    for metric, players in metrics.items():
        top_players = get_top_players(players, metric)
        print(f"{metric}:")
        for i, player in enumerate(top_players, start=1):
            print(f"{i}. {player['Player']} - {player[metric]}")
        print()

# %%
# Function to get the names of top N players based on a specific key
def get_top_player_names(dataset, key, n=40, reverse=True):
    top_players = get_top_players(dataset, key, n, reverse)
    return [player['Player'] for player in top_players]

# Get top 50 batsmen based on specified metrics
top_batsmen_names = get_top_player_names(cleaned_batting_stats, 'SR')  # Replace 'Strike Rate' with other metrics

print("Top 40 Batsmen:")
for i, player_name in enumerate(top_batsmen_names, start=1):
    print(f"{i}. {player_name}")

# Get top 50 bowlers based on specified metrics
top_bowlers_names = get_top_player_names(cleaned_bowling_stats, 'Wkts')  # Replace 'Wickets' with other metrics

print("\nTop 40 Bowlers:")
for i, player_name in enumerate(top_bowlers_names, start=1):
    print(f"{i}. {player_name}")


# %%
# Get top player names for all specified metrics
top_batsmen_names_strike_rate = get_top_player_names(cleaned_batting_stats, 'SR')
top_batsmen_names_hs = get_top_player_names(cleaned_batting_stats, 'HS')
top_batsmen_names_runs = get_top_player_names(cleaned_batting_stats, 'Runs')
top_batsmen_names_avg = get_top_player_names(cleaned_batting_stats, 'Avg')
top_batsmen_names_4s = get_top_player_names(cleaned_batting_stats, '4s')
top_batsmen_names_6s = get_top_player_names(cleaned_batting_stats, '6s')
top_batsmen_names_100 = get_top_player_names(cleaned_batting_stats, '100')

top_bowlers_names_wickets = get_top_player_names(cleaned_bowling_stats, 'Wkts')
top_bowlers_names_overs = get_top_player_names(cleaned_bowling_stats, 'Ov')
top_bowlers_names_economy = get_top_player_names(cleaned_bowling_stats, 'Econ')
top_bowlers_names_sr = get_top_player_names(cleaned_bowling_stats, 'SR')
top_bowlers_names_4w = get_top_player_names(cleaned_bowling_stats, '4w')
top_bowlers_names_5w = get_top_player_names(cleaned_bowling_stats, '5w')

# Combine all player names from different metrics
all_batsmen_names = set(top_batsmen_names_strike_rate + top_batsmen_names_hs +
                        top_batsmen_names_runs + top_batsmen_names_avg +
                        top_batsmen_names_4s + top_batsmen_names_6s +
                        top_batsmen_names_100)

all_bowlers_names = set(top_bowlers_names_wickets + top_bowlers_names_overs +
                        top_bowlers_names_economy + top_bowlers_names_sr +
                        top_bowlers_names_4w + top_bowlers_names_5w)

# Print all player names
print("All Top Batsmen Names:")
for i, player_name in enumerate(all_batsmen_names, start=1):
    print(f"{i}. {player_name}")

print("\nAll Top Bowlers Names:")
for i, player_name in enumerate(all_bowlers_names, start=1):
    print(f"{i}. {player_name}")

# %%
# Get the intersection of top batsmen and top bowlers to find all-rounders
all_rounders = set(all_batsmen_names).intersection(all_bowlers_names)

# Print the names of all-rounders
print("All-Rounders:")
for i, player_name in enumerate(all_rounders, start=1):
    print(f"{i}. {player_name}")

# %%
import matplotlib.pyplot as plt

# Function to generate and display performance graphs for a batsman
def plot_batsman_performance(batsman_name):
    # Retrieve data for the specified batsman
    batsman_data = [entry for entry in cleaned_batting_stats if entry['Player'] == batsman_name]
    
    if not batsman_data:
        print(f"No data found for {batsman_name}.")
        return
    
    # Extract relevant metrics for plotting
    years = [entry['Year'] for entry in batsman_data]
    strike_rates = [entry['SR'] for entry in batsman_data]
    
    # Create and display graphs
    plt.figure(figsize=(10, 6))
    
    plt.plot(years, strike_rates, marker='o', label='Strike Rate')
    
    # Plot other metrics
    
    plt.title(f"{batsman_name} - Performance Metrics")
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# applying the function on a single player to check the working 
plot_batsman_performance('Virat Kohli')

# %%
import matplotlib.pyplot as plt

# Function to generate and display performance graphs for a batsman
def plot_batsman_performance(batsman_name):
    # Retrieve data for the specified batsman
    batsman_data = [entry for entry in cleaned_batting_stats if entry['Player'] == batsman_name]
    
    if not batsman_data:
        print(f"No data found for {batsman_name}.")
        return
    
    # Extract relevant metrics for plotting
    years = [entry['Year'] for entry in batsman_data]
    strike_rates = [entry['SR'] for entry in batsman_data]
    hs_scores = [entry['HS'] for entry in batsman_data]
    runs = [entry['Runs'] for entry in batsman_data]
    avg = [entry['Avg'] for entry in batsman_data]
    fours = [entry['4s'] for entry in batsman_data]
    sixes = [entry['6s'] for entry in batsman_data]
    centuries = [entry['100'] for entry in batsman_data]
    
    # Create and display graphs
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 3, 1)
    plt.plot(years, strike_rates, marker='o')
    plt.title('Strike Rate')
    
    plt.subplot(3, 3, 2)
    plt.plot(years, hs_scores, marker='o')
    plt.title('Highest Score')
    
    plt.subplot(3, 3, 3)
    plt.plot(years, runs, marker='o')
    plt.title('Total Runs')
    
    plt.subplot(3, 3, 4)
    plt.plot(years, avg, marker='o')
    plt.title('Batting Average')
    
    plt.subplot(3, 3, 5)
    plt.plot(years, fours, marker='o')
    plt.title('4s')
    
    plt.subplot(3, 3, 6)
    plt.plot(years, sixes, marker='o')
    plt.title('6s')
    
    plt.subplot(3, 3, 7)
    plt.plot(years, centuries, marker='o')
    plt.title('Centuries')
    
    plt.tight_layout()
    plt.suptitle(f"{batsman_name} - Performance Metrics")
    plt.show()

# Apply the batsman performance plotting function to all top batsmen
# for batsman_name in all_batsmen_names:
# applying on a single player to check the working of the function
plot_batsman_performance('Virat Kohli')

# %%
import matplotlib.pyplot as plt

# Function to generate and display performance graphs for a bowler
def plot_bowler_performance(bowler_name):
    bowler_data = [entry for entry in cleaned_bowling_stats if entry['Player'] == bowler_name]
    
    if not bowler_data:
        print(f"No data found for {bowler_name}.")
        return
    
    # Extract relevant metrics for plotting
    years = [entry['Year'] for entry in bowler_data]
    wickets = [entry['Wkts'] for entry in bowler_data]
    overs = [entry['Ov'] for entry in bowler_data]
    economy = [entry['Econ'] for entry in bowler_data]
    sr = [entry['SR'] for entry in bowler_data]
    four_wickets = [entry['4w'] for entry in bowler_data]
    five_wickets = [entry['5w'] for entry in bowler_data]
    
    # Create and display graphs
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 3, 1)
    plt.plot(years, wickets, marker='o')
    plt.title('Wickets')
    
    plt.subplot(3, 3, 2)
    plt.plot(years, overs, marker='o')
    plt.title('Overs')
    
    plt.subplot(3, 3, 3)
    plt.plot(years, economy, marker='o')
    plt.title('Economy Rate')
    
    plt.subplot(3, 3, 4)
    plt.plot(years, sr, marker='o')
    plt.title('Strike Rate')
    
    plt.subplot(3, 3, 5)
    plt.plot(years, four_wickets, marker='o')
    plt.title('4-Wicket Hauls')
    
    plt.subplot(3, 3, 6)
    plt.plot(years, five_wickets, marker='o')
    plt.title('5-Wicket Hauls')
    
    plt.tight_layout()
    plt.suptitle(f"{bowler_name} - Performance Metrics")
    plt.show()

# Apply the bowler performance plotting function to all top bowlers
# for bowler_name in all_bowlers_names:
plot_bowler_performance('Rashid Khan')

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Function to perform linear regression and plot the trend for top 50 batsmen
def perform_linear_regression_for_top_batsmen(metric_name):
    # Get the names of top 50 batsmen
    top_batsmen_names = get_top_player_names(cleaned_batting_stats, metric_name, n=10)
    
    for batsman_name in top_batsmen_names:
        # Retrieve data for the specified batsman and metric
        batsman_data = [entry for entry in cleaned_batting_stats if entry['Player'] == batsman_name]
        
        # Extract relevant data for linear regression
        years = [entry['Year'] for entry in batsman_data]
        metric_data = [entry[metric_name] for entry in batsman_data]
        
        # Convert years to a 2D array
        X = np.array(years).reshape(-1, 1)
        
        # Create a linear regression model
        model = LinearRegression()
        model.fit(X, metric_data)
        
        # Predict the trend
        trend = model.predict(X)
        
        # Plot the data and the trend line
        plt.figure(figsize=(10, 6))
        plt.scatter(years, metric_data, color='blue', label='Actual Data')
        plt.plot(years, trend, color='red', label='Trend Line')
        plt.title(f"{batsman_name} - {metric_name} Trend Over Years")
        plt.xlabel("Year")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)
        plt.show()

# Perform linear regression and plot the trend for top 50 batsmen's strike rates
perform_linear_regression_for_top_batsmen('SR')

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Function to perform linear regression and plot the trend for top 50 bowlers
def perform_linear_regression_for_top_bowlers(metric_name):
    # Get the names of top 50 bowlers
    top_bowlers_names = get_top_player_names(cleaned_bowling_stats, metric_name, n=5)
    
    for bowler_name in top_bowlers_names:
        # Retrieve data for the specified bowler and metric
        bowler_data = [entry for entry in cleaned_bowling_stats if entry['Player'] == bowler_name]
        
        # Extract relevant data for linear regression
        years = [entry['Year'] for entry in bowler_data]
        metric_data = [entry[metric_name] for entry in bowler_data]
        
        # Convert years to a 2D array
        X = np.array(years).reshape(-1, 1)
        
        # Creating a linear regression model
        model = LinearRegression()
        model.fit(X, metric_data)
        
        # Predict the trend
        trend = model.predict(X)
        
        # Plot the data and the trend line
        plt.figure(figsize=(10, 6))
        plt.scatter(years, metric_data, color='blue', label='Actual Data')
        plt.plot(years, trend, color='red', label='Trend Line')
        plt.title(f"{bowler_name} - {metric_name} Trend Over Years")
        plt.xlabel("Year")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)
        plt.show()

# Performing linear regression and plot the trend for top 50 bowlers' wickets
perform_linear_regression_for_top_bowlers('Wkts')

# %%
# Function to construct a cricket team
def construct_cricket_team(top_batsmen, top_bowlers, all_rounders):
    team = {'Batsmen': [], 'Bowlers': [], 'All-Rounders': []}
    substitutes = {'Batsmen': [], 'Bowlers': [], 'All-Rounders': []}
    
    # Select batsmen with positive trend for the team
    for batsman_name in top_batsmen:
        # Check if the trend is positive (e.g., Strike Rate)
        if trend_is_positive(cleaned_batting_stats, batsman_name, 'SR'):
            if len(team['Batsmen']) < 4:
                team['Batsmen'].append(batsman_name)
            elif len(substitutes['Batsmen']) < 2:
                substitutes['Batsmen'].append(batsman_name)
    
    # Select bowlers with positive trend for the team
    for bowler_name in top_bowlers:
        # Check if the trend is positive (e.g., Wickets)
        if trend_is_positive(cleaned_bowling_stats, bowler_name, 'Wkts'):
            if len(team['Bowlers']) < 4:
                team['Bowlers'].append(bowler_name)
            elif len(substitutes['Bowlers']) < 1:
                substitutes['Bowlers'].append(bowler_name)
    
    # Select all-rounders with positive trend for the team
    for all_rounder_name in all_rounders:
        # Check if the trend is positive (e.g., Strike Rate or Wickets)
        if (trend_is_positive(cleaned_batting_stats, all_rounder_name, 'SR') or
            trend_is_positive(cleaned_bowling_stats, all_rounder_name, 'Wkts')):
            if len(team['All-Rounders']) < 3:
                team['All-Rounders'].append(all_rounder_name)
            elif len(substitutes['All-Rounders']) < 1:
                substitutes['All-Rounders'].append(all_rounder_name)
    
    return team, substitutes

# Function to check if a player's trend is positive
def trend_is_positive(dataset, player_name, metric_name):
    player_data = [entry for entry in dataset if entry['Player'] == player_name]
    metric_values = [entry[metric_name] for entry in player_data]
    
    if len(metric_values) < 2:
        return False
    
    diff = metric_values[-1] - metric_values[0]
    return diff > 0

# Get top 40 batsmen, bowlers, and all-rounders
top_batsmen = get_top_player_names(cleaned_batting_stats, 'SR', n=40)
top_bowlers = get_top_player_names(cleaned_bowling_stats, 'Wkts', n=40)
top_all_rounders = list(set(top_batsmen).intersection(top_bowlers))

# Constructing the cricket team
team, substitutes = construct_cricket_team(top_batsmen, top_bowlers, top_all_rounders)

# Printing the selected players (Batsman, Bowlers) in the team and substitutes
print("Team:")
print("Batsmen:", team['Batsmen'])
print("Bowlers:", team['Bowlers'])
print("All-Rounders:", team['All-Rounders'])

print("\nSubstitutes:")
print("Batsmen:", substitutes['Batsmen'])
print("Bowlers:", substitutes['Bowlers'])
print("All-Rounders:", substitutes['All-Rounders'])

# %%



