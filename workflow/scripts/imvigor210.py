

#!/usr/bin/env python3
"""
Example script for running floating feature selection.
"""

import sys
import os
import json
import pickle


print("Starting Floating Feature Selection Example Script")
# Add the package directory to Python path for direct import
package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'package'))
sys.path.insert(0, package_dir)

# Import directly from the ffs module
import ffs

save_results = True  # Set to True to save results to files
results_dir = 'results/Imvigor_FFS_Results'  # Directory to save results

datasets = [
    "integrated_DCB6",
    ]

for dataset in datasets:

    # Configuration
    run_id = 0              # Fixed seed for reproducibility
    #dataset = dataset #"recist_melanoma_cosmic_card_4-BINARY" #"madelon"  # Name of the dataset being used

    #data_path = "synthetic"  # Use an internal synthetic dataset (for development purposes)
    #data_path = "data/melanoma_batch_corrected_data_common_genes_cosmic_io.csv"


    if dataset == "integrated_DCB6":
        data_path = "data/imvigor/integrated_DCB6.h5ad"

    target_column =  "DCB6" #"target" #"recist"  # Specify the target column for real datasets

    n_experiments = 20   # Number of experiments to run

    all_results = {}
    for i in range(n_experiments):

        run_id = i   # Different seed for each experiment
        print(f"Running experiment {i+1}/{n_experiments} with run_id={run_id}")
        # Run the experiment    
        ffs_instance = ffs.FloatingFeatureSelector(run_id=run_id, data_path=data_path, target_column=target_column)
        experiment_result = ffs_instance.run_ffs(n_feat=10)  # Specify number of features to select before the optimization step

        print("Experiment completed successfully!")

        print("Selected features:", experiment_result)

        coverage = ffs_instance.Empirical_coverage_
        uncertainty = ffs_instance.Uncertainty_
        certainty = ffs_instance.Certainty_

        print(f"Empirical coverage: {coverage}")
        print(f"Uncertainty: {uncertainty}")
        print(f"Certainty: {certainty}")

        all_results[i+1] = {"selected_features": experiment_result, "run_id": i+1, "empirical_coverage": coverage,
                            "uncertainty": uncertainty, "certainty": certainty}



    print("All experiments completed!")
    print(all_results)

    print("Top selected features from all experiments:")
    # Count frequency of each feature across all experiments
    feature_counts = {}
    for experiment in all_results.values():
        for feature in experiment["selected_features"]:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

    # Sort features by frequency (descending)
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

    # Print top k features (e.g., top 10)
    k = 10
    print(f"Top {k} most selected features:")
    for i, (feature, count) in enumerate(sorted_features[:k], 1):
        print(f" {feature}: selected {count}/{n_experiments} times")

    # Save results to files
    
    
    if save_results:
        print("Saving results to files...")
        os.makedirs(results_dir, exist_ok=True)

        # Save as JSON (human-readable)
        json_file = os.path.join(results_dir, f"ffs_results_{dataset}_{n_experiments}.json")
        try:
            with open(json_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"Results saved to JSON: {json_file}")
        except Exception as e:
            print(f"Error saving JSON: {e}")

        # Save as pickle (preserves Python objects)
        pickle_file = os.path.join(results_dir, f"ffs_results_{dataset}_{n_experiments}.pkl")
        try:
            with open(pickle_file, 'wb') as f:
                pickle.dump(all_results, f)
            print(f"Results saved to pickle: {pickle_file}")
        except Exception as e:
            print(f"Error saving pickle: {e}")

        print("Results successfully saved!")




def analyze_ffs_results(json_file):
    from collections import Counter

    # Fix path issues for json_file
    json_file = os.path.normpath(json_file)
    print(f"Using JSON file: {json_file}")

    with open(json_file, "r") as f:
        ffs_results = json.load(f)
    # Extract all selected features from all runs
    all_features = []

    for run_data in ffs_results.values():
        # Parse the string representation of the array
        feature_str = run_data['selected_features']
        
        # Remove brackets and split by whitespace
        #features = feature_str.strip([]).split()
        # Convert to integers
        features = feature_str.strip().split()

        features = feature_str.replace('[', '').replace(']', '').split()
        feature_str = feature_str.replace('[', '').replace(']', '').split()
        features = [int(f) for f in feature_str]
        
        all_features.extend(features)

    # Count frequency of each feature
    feature_counts = Counter(all_features)

    # Get top k most frequent features
    k = 10  # You can change this value
    top_k_features = [feature for feature, count in feature_counts.most_common(k)]

    #print(f"Top {k} most frequent features:")
    #print(top_k_features)

    return top_k_features


top_k_features = analyze_ffs_results(json_file)
print("Top features:")
print(top_k_features)


  