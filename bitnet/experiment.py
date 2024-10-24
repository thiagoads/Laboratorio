import os
from pathlib import Path

import wandb
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from app.datasets import CustomDataset
from app.models import BaseModel, BitModel
from app.engine import train
from app.utils import plot_results, plot_decision_boundary

NUM_WORKERS = os.cpu_count()

WANDB_PROJECT_NAME = "bitnet"

def init_monitor(
        exp_id:str,
        model:torch.nn.Module,
        params:dict,
        tags:list=None):
    
    # 1. Start a new run
    run = wandb.init(project=WANDB_PROJECT_NAME, 
                     name = str(exp_id),
                     tags=tags)
    
    # 2. Save model inputs and hyperparameters
    config = run.config
    config.exp_id = exp_id
    config.model_name = model.name.lower()
    config.generator = str(params.generator)
    config.num_samples = params.num_samples
    config.num_features = params.num_features
    config.num_classes = params.num_classes
    config.hidden_layers = params.hidden_layers
    config.hidden_units = params.hidden_units
    config.batch_size = params.batch_size
    config.learning_rate = params.learning_rate
    config.epochs = params.epochs
    config.seed = params.seed
    config.device = params.device

    run.watch(model)
    return run


def start_experiment(
             exp_id:str=None,
             exp_path:Path=None,
             params:dict=None):
    

    print(f"Iniciamento execução do experimento: {exp_id}")
    
    DEFAULT_ACCURACY = Accuracy(task="multiclass", 
                            num_classes=params.num_classes)

    # gerando dados sintéticos c/ gerador escolhido

    X, y = params.generator.generate(num_classes=params.num_classes,
                              num_samples=params.num_samples,
                              num_features=params.num_features,
                              random_state=params.seed)

    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y)

    # divisão do conjunto de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=params.seed)

    # criando dataset c/ dados gerados
    train_dataset = CustomDataset(features = X_train, 
                                targets = y_train)

    test_dataset = CustomDataset(features = X_test, 
                                targets = y_test)


    # criando dataloader c/ dataset 
    train_dataloader = DataLoader(dataset = train_dataset, 
                                batch_size = params.batch_size, 
                                # reordena exemplos a cada época
                                shuffle = True,
                                num_workers = NUM_WORKERS)

    test_dataloader = DataLoader(dataset = test_dataset, 
                                batch_size = params.batch_size, 
                                shuffle = False, 
                                num_workers = NUM_WORKERS)

    # construção dos modelos
    base_model = BaseModel(input_size=params.num_features,
                        hidden_layers=params.hidden_layers, 
                        hidden_units=params.hidden_units, 
                        output_size=params.num_classes
                        ).to(params.device)

    bit_model = BitModel(input_size=params.num_features, 
                        hidden_layers=params.hidden_layers,
                        hidden_units=params.hidden_units, 
                        output_size=params.num_classes
                        ).to(params.device)

    # treinamento dos modelos
    criteria = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=base_model.parameters(), 
                                lr=params.learning_rate)
    

    # iniciando monitoramento do modelo base
    run = init_monitor(exp_id=exp_id,
                       model=base_model, 
                       params=params, 
                       tags=["baseline"])

    print("Treinando o modelo base")
    base_results = train(model = base_model, 
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    num_epochs=params.epochs,
                    criteria=criteria,
                    optimizer=optimizer,
                    metrics=DEFAULT_ACCURACY,
                    device = params.device,
                    progress=False,
                    output=True,
                    monitor=run)
    
    #finalizando monitor
    run.finish()


    criteria = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=bit_model.parameters(), 
                                lr=params.learning_rate)
    
    # inicinado monitoramento do modelo bitnet
    run = init_monitor(exp_id=exp_id,
                       model=bit_model, 
                       params=params, 
                       tags=["bitnet"])

    print("Treinando o modelo bitnet")
    bit_results = train(model = bit_model, 
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    num_epochs=params.epochs,
                    criteria=criteria,
                    optimizer=optimizer,
                    metrics=DEFAULT_ACCURACY,
                    device = params.device,
                    progress=False,
                    output=True,
                    monitor=run)
    
    #finalizando monitoramento
    run.finish()
    
    # salvando resultados dos treinamentos
    import json
    with open(f"{exp_path}/results/base.json", "w") as f:
        json.dump(base_results, f)
    with open(f"{exp_path}/results/bit.json", "w") as f:
        json.dump(bit_results, f)
    
    # salvando performance do treinamento
    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    plt.title(f"Performance do modelo Base")
    plot_results(results = base_results)
    plt.savefig(f"{exp_path}/images/base.png")
    plt.subplot(1, 2, 2)
    plt.title(f"Performance do modelo Bit")
    plot_results(results = bit_results)
    plt.savefig(f"{exp_path}/images/bit.png")


    # salvando decision boundaries
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.title(f"{base_model.name} Model")
    plot_decision_boundary(base_model, X_test, y_test)
    plt.subplot(1, 2, 2)
    plt.title(f"{bit_model.name} Model")
    plot_decision_boundary(bit_model, X_test, y_test)
    plt.savefig(f"{exp_path}/images/boundary.png")

    # salvando pesos do modelos
    torch.save(obj=base_model.state_dict(), f = f"{exp_path}/models/base.pt")
    torch.save(obj=bit_model.state_dict(), f = f"{exp_path}/models/bit.pt")