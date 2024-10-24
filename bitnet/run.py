import uuid
import json
import itertools
from pathlib import Path
from collections import namedtuple

import torch
from tqdm.auto import tqdm

from app.generators import LinearlySeparableDataGenerator, NonLinearlySeparableDataGenerator, MixedAndUnbalancedDataGenerator
from experiment import start_experiment

def get_generators(names: list):
    generators = []
    for name in names:
        if(name == "linear"):
            generators.append(LinearlySeparableDataGenerator())
        if(name == "nonlinear"):
            generators.append(NonLinearlySeparableDataGenerator())    
        if(name == "unbalanced"):
            generators.append(MixedAndUnbalancedDataGenerator())
    return generators

def get_devices(names: list):
    devices = []
    for name in names:
        if name == "cuda" and torch.cuda.is_available():
            devices.append("cuda")
        if name == "cpu":
            devices.append("cpu")
    return devices

def get_grid(config):
    GENERATORS = get_generators(config["params"]["generators"])
    NUM_SAMPLES = config["params"]["num_samples"]
    NUM_CLASSES = config["params"]["num_classes"]
    NUM_FEATURES = config["params"]["num_features"]
    BATCH_SIZES = config["params"]["batch_sizes"]
    HIDDEN_LAYERS = config["params"]["hidden_layers"]
    HIDDEN_UNITS = config["params"]["hidden_units"]
    LEARNING_RATES = config["params"]["learning_rates"]
    EPOCHS = config["params"]["epochs"]
    SEEDS = config["params"]["seeds"]
    DEVICES = get_devices(config["params"]["devices"])
    # retorna tuplas com a combinação de todas as possibilidades
    return itertools.product(GENERATORS, NUM_SAMPLES, NUM_CLASSES, NUM_FEATURES, 
                             BATCH_SIZES, HIDDEN_LAYERS, HIDDEN_UNITS, LEARNING_RATES, 
                             EPOCHS, SEEDS, DEVICES)

# cria diretorios necessários
def dirs(config):
    exp_dir = Path(config["exp_dir"])

    exp_dir.mkdir(parents=True,
                  exist_ok=True)
    
def exps(config):
    exp_id = uuid.uuid4()
    exp_path = Path(config["exp_dir"] + f"/{exp_id}")
    exp_path.mkdir(parents=True, exist_ok=True)
    exp_results = Path(config["exp_dir"] + f"/{exp_id}/results")
    exp_results.mkdir(parents=True, exist_ok=True)
    exp_images = Path(config["exp_dir"] + f"/{exp_id}/images")
    exp_images.mkdir(parents=True, exist_ok=True)
    exp_models = Path(config["exp_dir"] + f"/{exp_id}/models")
    exp_models.mkdir(parents=True, exist_ok=True)
    return exp_id, exp_path

# inicia execução dos experimentos     
def main(config):

    combinations = get_grid(config)
    myexp_params = [ "generator", "num_samples", "num_classes", "num_features", 
                    "batch_size", "hidden_layers", "hidden_units", "learning_rate", 
                    "epochs", "seed", "device"]
    parameters = namedtuple("params", myexp_params)
    
    # Loop through each combination
    for combination in combinations:
        # criando diretório p/ resultados do experimento
        exp_id, exp_path = exps(config)

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
        
        