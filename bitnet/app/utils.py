
import torch
import matplotlib.pyplot as plt
import numpy as np

# Source https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py
# Com modificações p/ esperar y sempre em One Hot Encoding
def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # ajuste para considerar adicionar número de colunas esperadas pelo modelo
    extra_columns = len(X[0]) - 2 
    if extra_columns > 0:
        X_with_extra_columns = np.zeros((X_to_pred_on.shape[0], 2 + extra_columns), dtype=np.float32)
        X_with_extra_columns[:, :-extra_columns] = X_to_pred_on
        X_to_pred_on = torch.from_numpy(X_with_extra_columns)

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Modificação p/ sempre tratar y_pred como vector
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def plot_results(results: dict):
    fig, (ax1, ax2) = plt.subplots(nrows=1, 
                                   ncols=2, 
                                   sharex=True, 
                                   figsize=(9, 4))
    fig.suptitle(f"Performance do modelo {results['model_name']}",
                 fontsize="x-large")
    ax1.plot(results["train_loss"], label = "Train")
    ax1.plot(results["test_loss"], label = "Test")
    ax1.set_title("Loss", fontsize = 'large')
    ax1.set_xlabel("Epochs")
    ax2.plot(results["train_acc"], label = "Train")
    ax2.plot(results["test_acc"], label = "Test")
    ax2.set_title("Accuracy", fontsize = 'large')
    ax2.set_xlabel("Epochs")
    plt.legend()