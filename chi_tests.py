from scipy.stats import chisquare
from main import main

results_df = main()

# set draw dates to shape of results_df
num_draw_dates = results_df.shape[0]

# Get counts for all white balls
white_ball_counts = results_df['white_balls'].explode().value_counts().to_dict()

# Top 10 white balls
white_ball_top10 = list(white_ball_counts.keys())[:10]

# Frequencies for the top 10 white balls
white_ball_observed = [white_ball_counts[ball] for ball in white_ball_top10]

# Extend white_ball_observed list for all balls
all_white_balls = [i for i in range(1, 70)]
for ball in all_white_balls:
    if ball not in white_ball_top10:
        white_ball_observed.append(white_ball_counts.get(ball, 0))

# Compute expected frequencies for white balls
total_white_balls_drawn = sum(white_ball_observed)
white_ball_expected = [total_white_balls_drawn / len(all_white_balls) for _ in all_white_balls]

# Chi-squared test
white_ball_chi2, white_ball_p = chisquare(white_ball_observed, f_exp=white_ball_expected)

print("White Ball Chi-squared Test:")
print(f"Chi2 Statistic: {white_ball_chi2}")
print(f"P-value: {white_ball_p}")


# Get counts for all powerballs
powerball_counts = results_df['powerball'].value_counts().to_dict()

# Top 10 powerballs
powerball_top10 = list(powerball_counts.keys())[:10]

# Frequencies for the top 10 powerballs
powerball_observed = [powerball_counts[ball] for ball in powerball_top10]

# Extend powerball_observed list for all balls
all_powerballs = [i for i in range(1, 27)]  # Adjust the range if the range of your powerball numbers is different
for ball in all_powerballs:
    if ball not in powerball_top10:
        powerball_observed.append(powerball_counts.get(ball, 0))

# Compute expected frequencies for powerballs
total_powerballs_drawn = sum(powerball_observed)
powerball_expected = [total_powerballs_drawn / len(all_powerballs) for _ in all_powerballs]

# Chi-squared test
powerball_chi2, powerball_p = chisquare(powerball_observed, f_exp=powerball_expected)

print("\nPowerball Chi-squared Test:")
print(f"Chi2 Statistic: {powerball_chi2}")
print(f"P-value: {powerball_p}")


