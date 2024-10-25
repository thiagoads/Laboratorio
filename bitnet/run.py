import uuid
import json
import random
import itertools
from pathlib import Path
from datetime import datetime
from collections import namedtuple

import torch
from tqdm.auto import tqdm

from app.generators import LinearlySeparableDataGenerator, NonLinearlySeparableDataGenerator, MixedAndUnbalancedDataGenerator
from experiment import start_experiment

def filter_devices(names: list):
    devices = []
    for name in names:
        if name == "cuda" and torch.cuda.is_available():
            devices.append("cuda")
        if name == "cpu":
            devices.append("cpu")
    return devices

def get_combinations(config):
    generators = config["params"]["generators"]
    num_samples = config["params"]["num_samples"]
    num_classes = config["params"]["num_classes"]
    num_features = config["params"]["num_features"]
    batch_sizes = config["params"]["batch_sizes"]
    hidden_layers = config["params"]["hidden_layers"]
    hidden_units = config["params"]["hidden_units"]
    learning_rates = config["params"]["learning_rates"]
    epochs = config["params"]["epochs"]
    seeds = config["params"]["seeds"]
    activations = config["params"]["activations"]
    devices = filter_devices(config["params"]["devices"])

    # retorna tuplas com a combinação de todas as possibilidades
    combinations = list(itertools.product(generators, num_samples, num_classes, num_features, 
                            batch_sizes, hidden_layers, hidden_units, learning_rates, 
                            activations, epochs, seeds, devices))
    method = config["method"]
    if method == "grid":
        # return all possible combinations (this can take too much time!)
        return combinations
    
    if method == "random":
        # Calculate the number of samples to select (minimum of 1000 or 10% of the total)
        num_samples = min(1000, int(0.1 * len(combinations)))
        # Randomly select the specified number of combinations
        random_combinations = random.sample(combinations, num_samples)
        return random_combinations
    
    raise ValueError(f"Method {method} not supported or specified! Choose between 'grid' or 'random'.")

# cria diretorios necessários
def dirs(config):
    exp_dir = Path(config["exp_dir"])

    exp_dir.mkdir(parents=True,
                  exist_ok=True)
    
def exps(run_dir):
    exp_id = uuid.uuid4()
    exp_path = Path(run_dir + f"/{exp_id}")
    exp_path.mkdir(parents=True, exist_ok=True)
    exp_results = Path(run_dir + f"/{exp_id}/results")
    exp_results.mkdir(parents=True, exist_ok=True)
    exp_images = Path(run_dir + f"/{exp_id}/images")
    exp_images.mkdir(parents=True, exist_ok=True)
    exp_models = Path(run_dir + f"/{exp_id}/models")
    exp_models.mkdir(parents=True, exist_ok=True)
    exp_params = Path(run_dir + f"/{exp_id}/params")
    exp_params.mkdir(parents=True, exist_ok=True)
    return exp_id, exp_path

# inicia execução dos experimentos     
def main(config):
    now_str = datetime.now().strftime("%Y%m%d.%H%M%S.%f")
    run_dir = config["exp_dir"] + "/" + now_str

    combinations = get_combinations(config)

    params_keys = [ "generator", "num_samples", "num_classes", "num_features", 
                    "batch_size", "hidden_layers", "hidden_units", "learning_rate", 
                    "activation_alias", "epochs", "seed", "device"]
    
    parameters = namedtuple("params", params_keys)


    rounds = config["rounds"]
    for run in range(rounds):

        print("+------------------------------------------------------------+")
        print(f" Rodada #{run+1}      | N. Experimentos: {len(combinations)} ")
        # Loop through each combination
        for index, combination in enumerate(combinations):
            # criando diretório p/ resultados do experimento
            exp_id, exp_path = exps(run_dir)

            # extraindo combinação de parametros
            params = parameters(*combination)
            print("+------------------------------------------------------------+")
            print(f" Experimento #{index + 1} | Id: {exp_id}                     ")
            print("+------------------------------------------------------------+")
            print(params)
            print("+------------------------------------------------------------+")

            # iniciando treinamento 
            start_experiment(
                    exp_id=exp_id,
                    exp_path=exp_path,
                    params=params)
        

if __name__ == "__main__":

    with open("config.json", "r") as f:
        config = json.load(f)
    
    dirs(config=config)
    main(config=config)
        
        