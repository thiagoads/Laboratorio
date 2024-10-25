import random
from itertools import product
import pprint

# Define the parameters
params = {
    "seeds": [None],
    "num_samples": [1000, 2000],
    "num_classes": [2, 3],
    "num_features": [2, 4],
    "batch_sizes": [32, 1],
    "hidden_layers": [2, 4, 8],
    "hidden_units": [5, 50, 100],
    "epochs": [1, 5, 10],
    "learning_rates": [1e-3, 1e-4],
    "activations": ["tanh", "relu", "selu"],
    "generators": ["linear", "nonlinear", "unbalanced"],
    "devices": ["cuda"]
}

# Get all keys and values from the parameters dictionary
keys, values = params.keys(), params.values()

# Use itertools.product to calculate all combinations
combinations = list(product(*values))

# Calculate the number of samples to select (minimum of 1000 or 10% of the total)
num_samples = min(1000, int(0.1 * len(combinations)))

# Randomly select the specified number of combinations
random_combinations = random.sample(combinations, num_samples)

# Format the selected combinations into a list of dictionaries
combination_dicts = [dict(zip(keys, combination)) for combination in random_combinations]

# Print the first few combinations for preview (since the total might be large)
pprint.pprint(combination_dicts[:5])
print(f"\nTotal selected combinations: {len(combination_dicts)}")
