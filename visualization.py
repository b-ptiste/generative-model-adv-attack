import json
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader


def plot_loss(epoch_losses: list) -> None:
    """plot loss

    Args:
        epoch_losses (list): list of loss
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses, label="Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluation_vizualization(
    all_preds: list, all_targets: list, num_classes: int
) -> tuple:
    """evaluation vizualization

    Args:
        all_preds (list): list of predictions
        all_targets (list): list of targets
        num_classes (int): number of classes

    Returns:
        tuple: accuracy, precision, recall, f1
    """
    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average="macro")
    recall = recall_score(all_targets, all_preds, average="macro")
    f1 = f1_score(all_targets, all_preds, average="macro")
    conf_matrix = confusion_matrix(all_targets, all_preds)

    # Plot Confusion Matrix with percentages
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    # Adding text annotations with percentages
    total = conf_matrix.sum()
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j,
                i,
                f"{conf_matrix[i, j]}\n({conf_matrix[i, j] / total:.2%})",
                horizontalalignment="center",
                color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black",
            )

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    # Print metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    return accuracy, precision, recall, f1


def load_result(path: str) -> dict:
    """load result

    Args:
        path (str): path to result

    Returns:
        dict: result
    """
    keys_order = ["GFZ", "GBZ", "GFY", "GBY", "DBX", "DFX", "DFZ"]

    with open(path, "r") as f:
        loaded_dict = json.load(f)

    loaded_dict = {key: loaded_dict[key] for key in keys_order if key in loaded_dict}
    return loaded_dict


def visualisation_line_plot(
    loaded_dict: dict, metrics: list, epsilon_values: list
) -> None:
    """visualisation line plot

    Args:
        loaded_dict (dict): result
        metrics (list): metrics
        epsilon_values (list): epsilon values
    """
    # Preparing plot data
    plot_data = {
        model: {metric: [] for metric in metrics} for model in loaded_dict.keys()
    }
    for model, model_data in loaded_dict.items():
        for epsilon in epsilon_values:
            for metric in metrics:
                plot_data[model][metric].append(model_data[epsilon][metric])
    palette = sns.color_palette("husl", len(loaded_dict))

    sns.set(style="darkgrid")
    fig, axes = plt.subplots(1, 4, figsize=(30, 6))
    fig.suptitle(
        "Model Performance Across Different $\epsilon$ Values",
        fontsize=20,
        fontweight="bold",
    )

    for i, metric in enumerate(metrics):
        ax = axes[i % 4]
        for j, (model, model_data) in enumerate(plot_data.items()):
            sns.lineplot(
                x=epsilon_values,
                y=model_data[metric],
                marker="o",
                label=model,
                ax=ax,
                color=palette[j],
                linewidth=2.5,
            )
        ax.set_title(f"{metric.capitalize()} vs $\epsilon$", fontsize=16)
        ax.set_xlabel("$\epsilon$", fontsize=14)
        ax.set_ylabel(metric.capitalize(), fontsize=14)
        ax.legend(title="Models", title_fontsize="13", fontsize="12")
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.spines["top"].set_color("grey")
        ax.spines["right"].set_color("grey")
        ax.tick_params(axis="both", which="major", labelsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


def visualisation_heatmap(loaded_dict: dict, epsilon_values: list) -> None:
    """visualisation heatmap

    Args:
        loaded_dict (dict): result
        epsilon_values (list): epsilon values
    """

    accuracy_values = {epsilon: [] for epsilon in epsilon_values}

    models = list(loaded_dict.keys())

    for epsilon in epsilon_values:
        for model in models:
            accuracy_values[epsilon].append(loaded_dict[model][epsilon]["accuracy"])
    print(models)
    # Creating a dataframe for the heatmap
    heatmap_data = np.array(list(accuracy_values.values())).T
    heatmap_df = pd.DataFrame(heatmap_data, index=models, columns=epsilon_values)

    # Creating the heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
    plt.title("Accuracy of Models Across Different $\epsilon$ Values")
    plt.ylabel("Models")
    plt.xlabel("$\epsilon$")

    plt.show()


def display_random_images(
    dataloader: DataLoader,
    num_to_class: int = None,
    dataset_name: str = "FASHION_MNIST",
) -> None:
    """_summary_

    Args:
        dataloader (DataLoader): Dataloader
        num_to_class (int, optional): Number of classes. Defaults to None.
        dataset_name (str, optional): Dataset name. Defaults to "FASHION_MNIST".
    """
    # Retrieve a single batch from the dataloader
    images, labels = next(iter(dataloader))

    chosen_indices = random.sample(range(len(images)), 5)

    # Plotting
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.tight_layout(pad=3.0)

    for i, idx in enumerate(chosen_indices):
        img = images[idx]
        label = labels[idx]

        if dataset_name == "SVHN":
            img = img * 0.5 + 0.5  # Reverse the normalization

        img = img.numpy().transpose((1, 2, 0))

        class_name = (
            label.item()
            if dataset_name != "FASHION_MNIST"
            else num_to_class[label.item()]
        )
        axes[i].imshow(img, cmap="gray" if dataset_name != "SVHN" else None)
        axes[i].set_title(class_name)
        axes[i].axis("off")

    plt.show()
