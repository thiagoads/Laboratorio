import os

import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from app.generators import LinearlySeparableDataGenerator, NonLinearlySeparableDataGenerator, MixedAndUnbalancedDataGenerator
from app.datasets import CustomDataset
from app.models import BaseModel, BitModel
from app.engine import train
from app.utils import plot_results, plot_decision_boundary

DEFAULT_SEED = 42
NUM_SAMPLES = 1000
NUM_CLASSES = 2
NUM_FEATURES = 2
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
HIDDEN_UNITS = 5
EPOCHS = 10
LEARNING_RATE = 1e-3
SYNTHETIC_DATA_GENERATOR = LinearlySeparableDataGenerator()
DEFAULT_ACCURACY = Accuracy(task="multiclass", 
                            num_classes=NUM_CLASSES)

if __name__ == "__main__":

    # selecionado device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")


    # gerando dados sintéticos c/ gerador escolhido
    print("Gerando dados sintéticos:")
    print(f"Gerador: {SYNTHETIC_DATA_GENERATOR}")
    print(f"Features: {NUM_FEATURES}")
    print(f"Classes: {NUM_CLASSES}")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"Semente: {DEFAULT_SEED}")
    X, y = SYNTHETIC_DATA_GENERATOR.generate(num_classes=NUM_CLASSES,
                                            num_samples=NUM_SAMPLES,
                                            num_features=NUM_FEATURES,
                                            random_state=DEFAULT_SEED)

    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y)

    # divisão do conjunto de treinamento e teste
    print("Divisão dos conjuntos de treinamento e teste")
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=DEFAULT_SEED)

    # criando dataset c/ dados gerados
    print("Criando datasets")
    train_dataset = CustomDataset(features = X_train, 
                                targets = y_train)

    test_dataset = CustomDataset(features = X_test, 
                                targets = y_test)


    # criando dataloader c/ dataset 
    print("Criando dataloader de treinamento e teste" )
    print(f"Batch: {BATCH_SIZE}")
    train_dataloader = DataLoader(dataset = train_dataset, 
                                batch_size = BATCH_SIZE, 
                                # reordena exemplos a cada época
                                shuffle = True,
                                num_workers = NUM_WORKERS)

    test_dataloader = DataLoader(dataset = test_dataset, 
                                batch_size = BATCH_SIZE, 
                                shuffle = False, 
                                num_workers = NUM_WORKERS)


    # construção dos modelos
    print("Criação dos modelos")
    print(f"Input: {NUM_FEATURES}")
    print(f"Units: {HIDDEN_UNITS}")
    print(f"Output: {NUM_CLASSES}")
    base_model = BaseModel(input_size=NUM_FEATURES, 
                        hidden_units=HIDDEN_UNITS, 
                        output_size=NUM_CLASSES
                        ).to(device)

    bit_model = BitModel(input_size=NUM_FEATURES, 
                        hidden_units=HIDDEN_UNITS, 
                        output_size=NUM_CLASSES
                        ).to(device)



    # treinamento dos modelos
    print("Parâmetros de treinamento")
    print(f"Métrica: {DEFAULT_ACCURACY}")
    print(f"Taxa (LR): {LEARNING_RATE}")
    print(f"Épocas: {EPOCHS}")

    criteria = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=base_model.parameters(), 
                                lr=LEARNING_RATE)


    print("Iniciando treinamento do modelo Base")
    base_results = train(model = base_model, 
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    num_epochs=EPOCHS,
                    criteria=criteria,
                    optimizer=optimizer,
                    metrics=DEFAULT_ACCURACY,
                    device = device)

    criteria = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=bit_model.parameters(), 
                                lr=LEARNING_RATE)

    print("Iniciando treinamento do modelo BitNet")
    bit_results = train(model = bit_model, 
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    num_epochs=EPOCHS,
                    criteria=criteria,
                    optimizer=optimizer,
                    metrics=DEFAULT_ACCURACY,
                    device = device)

    print("Exibindo resultados do treinamento")
    # plotando performance do treinamento
    plot_results(results = base_results)
    plot_results(results = bit_results)

    # plotando decision boundaries
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.title(f"{base_model.name} Model")
    plot_decision_boundary(base_model, X_test, y_test)
    plt.subplot(1, 2, 2)
    plt.title(f"{bit_model.name} Model")
    plot_decision_boundary(bit_model, X_test, y_test)