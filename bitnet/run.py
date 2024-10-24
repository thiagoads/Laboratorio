import uuid
import json
import itertools, more_itertools
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
    GENERATORS = config["params"]["generators"]
    NUM_SAMPLES = config["params"]["num_samples"]
    NUM_CLASSES = config["params"]["num_classes"]
    NUM_FEATURES = config["params"]["num_features"]
    BATCH_SIZES = config["params"]["batch_sizes"]
    HIDDEN_LAYERS = config["params"]["hidden_layers"]
    HIDDEN_UNITS = config["params"]["hidden_units"]
    LEARNING_RATES = config["params"]["learning_rates"]
    EPOCHS = config["params"]["epochs"]
    SEEDS = config["params"]["seeds"]
    ACTIVATIONS = config["params"]["activations"]
    DEVICES = filter_devices(config["params"]["devices"])

    method = config["method"]
    if method == "grid":
        # retorna tuplas com a combinação de todas as possibilidades
        return itertools.product(GENERATORS, NUM_SAMPLES, NUM_CLASSES, NUM_FEATURES, 
                                BATCH_SIZES, HIDDEN_LAYERS, HIDDEN_UNITS, LEARNING_RATES, 
                                ACTIVATIONS, EPOCHS, SEEDS, DEVICES)
    if method == "random":
        # retorna tuplas com a combinação de todas as possibilidades randdomizadas
        return more_itertools.random_product(GENERATORS, NUM_SAMPLES, NUM_CLASSES, NUM_FEATURES, 
                                BATCH_SIZES, HIDDEN_LAYERS, HIDDEN_UNITS, LEARNING_RATES, 
                                ACTIVATIONS, EPOCHS, SEEDS, DEVICES)
    
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
    myexp_params = [ "generator", "num_samples", "num_classes", "num_features", 
                    "batch_size", "hidden_layers", "hidden_units", "learning_rate", 
                    "activation_alias", "epochs", "seed", "device"]
    parameters = namedtuple("params", myexp_params)
    
    # Loop through each combination
    for combination in combinations:
        # criando diretório p/ resultados do experimento
        exp_id, exp_path = exps(run_dir)

        # extraindo combinação de parametros
        params = parameters(*combination)
        
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
        
        