import itertools
import pandas as pd
from sodapy import Socrata
from predictive_analysis import run_predictive_analysis


def main():
    # Set up an authenticated client
    client = Socrata("data.ny.gov",
                     "",    # app token
                     username="",
                     password="")

    # Fetch all results in batches
    results = []
    offset = 0
    batch_size = 1000
    while True:
        batch = client.get("d6yy-54nr", limit=batch_size, offset=offset)
        if not batch:
            break
        results.extend(batch)
        offset += batch_size

    # Checking results before pandas interaction
    # After fetching the data from the API:
    print("First few raw JSON results:")
    for i in range(min(5, len(results))):
        print(results[i])
    print("\n")

    # Convert results to pandas DataFrame
    results_df = pd.DataFrame.from_records(results)

    # Convert the 'winning_numbers' strings into lists of numbers
    results_df['numbers_list'] = results_df['winning_numbers'].str.split().apply(lambda x: [int(num) for num in x])

    # Separate white ball numbers and the Powerball
    results_df['white_balls'] = results_df['numbers_list'].apply(lambda x: x[:-1])
    results_df['powerball'] = results_df['numbers_list'].apply(lambda x: x[-1])

    # Count the number of draw dates
    num_draw_dates = results_df.shape[0]

    print(f"Analyzing {num_draw_dates} draw dates")
    print("\n")  # for some spacing

    # Analyzing Common White Ball Numbers
    print("Top 10 Most Common White Ball Numbers:")
    common_white_balls = results_df['white_balls'].explode().value_counts().head(10)
    for number, count in common_white_balls.items():
        print(f"Number: {number}, Count: {count}")

    print("\n")

    # Analyzing Common Powerball Numbers
    print("Top 10 Most Common Powerball Numbers:")
    common_powerballs = results_df['powerball'].value_counts().head(10)
    for number, count in common_powerballs.items():
        print(f"Number: {number}, Count: {count}")

    # Combination Analysis
    # Convert the 'white_balls' lists into tuples
    results_df['white_ball_combos'] = results_df['white_balls'].apply(tuple)

    # Count the occurrences of each combination
    common_combos = results_df['white_ball_combos'].value_counts().head(10)

    # Print the results
    print("\nTop 10 Most Common White Ball Combinations:")
    for combo, count in common_combos.items():
        print(f"Combination: {' '.join(map(str, combo))}, Count: {count}")

    # Filter the combinations that appear more than once
    repeated_combos = common_combos[common_combos > 1]

    if not repeated_combos.empty:
        print("\nRepeated White Ball Combinations:")
        for combo, count in repeated_combos.items():
            print(f"Combination: {' '.join(map(str, combo))}, Count: {count}")
    else:
        print("\nNo white ball combinations have been repeated.")

    # Get top 10 most common white balls
    top_white_balls = results_df['white_balls'].explode().value_counts().head(10).index.tolist()
    # Generate combinations of 5 white balls
    white_ball_combinations = list(itertools.combinations(top_white_balls, 5))

    # Get top 10 most common powerballs
    top_powerballs = results_df['powerball'].value_counts().head(10).index.tolist()

    # Get the first 10 combinations pairing with the most common powerball
    predictive_combinations = []
    for i in range(10):
        white_combo = white_ball_combinations[i]
        powerball = top_powerballs[i % len(top_powerballs)]  # Cycle through the top powerballs
        predictive_combinations.append((*white_combo, powerball))

    # Print the "predictive" combinations
    print("\nPredictive Winning Combinations Based on Historical Frequency:")
    for combo in predictive_combinations:
        print(' '.join(map(str, combo[:-1])) + f" Powerball: {combo[-1]}")

    print("\nPredictive Winning Combinations Analysis Using scikit-learn:")
    run_predictive_analysis(results_df, white_ball_combinations)

    return results_df


if __name__ == "__main__":
    main()





