from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def tune_hyperparameters(X_train, y_train, X_test, y_test):
    # Step 1: Define the hyperparameter grid
    param_grid = {
        'n_estimators': [10, 50, 100, 150, 200],  # number of trees in the forest
        'max_depth': [None, 10, 20, 30, 40, 50],  # maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4]  # minimum number of samples required to be at a leaf node
    }

    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier()

    # Step 2: Use GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)

    # Step 3: Fit and evaluate
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_

    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    # Evaluate on the test set
    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy on Test Set:", accuracy)

    return best_rf  # return the best model for potential future use


def run_predictive_analysis(results_df, white_ball_combinations):
    results_df['avg_white_balls'] = results_df['white_balls'].apply(lambda x: sum(x)/len(x))
    results_df['sum_white_balls'] = results_df['white_balls'].apply(sum)
    results_df['last_powerball'] = results_df['powerball'].shift(1)

    # Dropping any rows with NaN values
    results_df.dropna(inplace=True)

    # Splitting data into features (X) and target (y)
    X = results_df[['avg_white_balls', 'sum_white_balls', 'last_powerball']]
    y = results_df['powerball']

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tuning hyperparameters and obtaining the best Random Forest model
    best_rf = tune_hyperparameters(X_train, y_train, X_test, y_test)

    # Making predictions for 10 potential Powerball numbers using the best model
    powerball_predictions = best_rf.predict(X_test[:10])  # We just use the first 10 test samples

    # Pair these predictions with the top 10 most common white ball combinations
    print("\nPredictive Winning Combinations Based on ML Model:")
    for i in range(10):
        white_combo = white_ball_combinations[i]
        print(f"{' '.join(map(str, white_combo))} Powerball: {powerball_predictions[i]}")

    # Make predictions using the best_rf model for evaluation
    best_rf_predictions = best_rf.predict(X_test)

    # Print a sample of the best_rf model's predictions for the test set
    print("\nModel Predictions (Sample for the Test Set):")
    for i, pred in enumerate(best_rf_predictions[:10]):  # just the first 10 for brevity
        print(f"Prediction {i + 1}: {pred}")
    print("...\n")

    # Printing the accuracy in a more human-readable format
    accuracy = accuracy_score(y_test, best_rf_predictions)
    print(f"Accuracy: {accuracy:.2%}\n")

    # Printing a concise classification report
    report = classification_report(y_test, best_rf_predictions, zero_division=1, output_dict=True)
    print(f"Macro Average Precision: {report['macro avg']['precision']:.2f}")
    print(f"Macro Average Recall: {report['macro avg']['recall']:.2f}")
    print(f"Macro Average F1-Score: {report['macro avg']['f1-score']:.2f}")
    print(f"Weighted Average Precision: {report['weighted avg']['precision']:.2f}")
    print(f"Weighted Average Recall: {report['weighted avg']['recall']:.2f}")
    print(f"Weighted Average F1-Score: {report['weighted avg']['f1-score']:.2f}\n")
