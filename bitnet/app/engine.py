import torch
from tqdm.auto import tqdm
from torchmetrics.classification import Accuracy


def train_step(model: torch.nn.Module,
          dataloader: torch.utils.data.DataLoader,
          criteria: torch.nn.Module,
          metrics: Accuracy,
          optimizer: torch.optim.Optimizer,
          device: torch.device):
    
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        y_logits = model(X)

        loss = criteria(y_logits, y)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        acc = metrics(torch.argmax(y_logits, dim = 1).cpu(), y.cpu())
        train_acc += acc.item()

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              criteria: torch.nn.Module, 
              metrics: Accuracy, 
              device: torch.device):
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for test_batch, (X_test, y_test) in enumerate(dataloader):
            X_test, y_test = X_test.to(device), y_test.to(device)

            y_logits_test = model(X_test.to(device))
            
            test_loss += criteria(y_logits_test, y_test).item()
            test_acc += metrics(torch.argmax(y_logits_test, dim = 1).cpu(), y_test.cpu()).item()

    test_loss = test_loss / len(dataloader)    
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc



def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          num_epochs: int,
          criteria: torch.nn.Module,
          metrics: Accuracy,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          progress = True,
          output = True,
          monitor = None):
    
    results = {
        "model_name": str(model.name),
        "train_loss": [], "train_acc": [],  
        "test_loss": [], "test_acc": []
    }

    model.to(device)

    # començando épocas de treinamento do modelo
    for epoch in tqdm(range(num_epochs), disable=(not progress)):


        # treinamento do modelo por uma época
        train_loss, train_acc = train_step(model = model,
                                           dataloader = train_dataloader,
                                           criteria = criteria,
                                           metrics = metrics,
                                           optimizer = optimizer,
                                           device = device)

        # avaliando modelo nos dados de teste
        test_loss, test_acc = test_step(model = model, 
                                        dataloader = test_dataloader, 
                                        criteria = criteria, 
                                        metrics = metrics, 
                                        device = device)
        
        if monitor:
            # 4. Log metrics to visualize performance
            monitor.log({"train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc})

        if output:
            # exibindo métricas de treinamento e teste
            print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

