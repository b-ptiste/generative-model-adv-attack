import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm


from src.visualization import evaluation_vizualization, plot_loss


def train(
    trainloader: DataLoader,
    lr: float,
    num_epochs: int,
    num_classes: int,
    latent_size: int,
    H: int,
    name_model: str = "GFZ",
    dataset: str = "MNIST",
) -> tuple[nn.Module, torch.Tensor]:
    """train a CVAE model

    Args:
        trainloader (DataLoader): train DataLoader
        lr (float): learning rate
        num_epochs (int): number of epochs
        num_classes (int): number of classes
        latent_size (int): latent size
        H (int): hidden size
        name_model (str, optional): name of the models. Defaults to "GFZ".
        dataset (str, optional): name of the dataset. Defaults to "MNIST".

    Returns:
        tuple[nn.Module, torch.Tensor]: return the model and the loss
    """
    print(f"Train on {dataset}")
    if "MNIST" in dataset:
        from model_MNIST import CVAE
    elif dataset == "SVHN":
        from model_SVHN import CVAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cvae = CVAE(latent_size, num_classes, H, name_model).to(device)

    # Optimizer
    optimizer = optim.Adam(cvae.parameters(), lr=lr)

    # Training CVAE
    cvae.train()
    print(f"Using device: {device}")

    cvae.to(device)
    epoch_losses = []
    for epoch in range(num_epochs):
        train_loader_tqdm = tqdm(
            enumerate(trainloader),
            total=len(trainloader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
        )

        train_loss = 0
        it = 1
        for batch_idx, (data, target) in train_loader_tqdm:
            data, target = data.to(device, dtype=torch.float32), target.to(device)
            target = torch.nn.functional.one_hot(target, num_classes=num_classes).to(
                torch.float32
            )
            optimizer.zero_grad()

            # Calculate loss
            loss = -cvae(data, target, it)
            loss = loss.mean()
            epoch_losses.append(loss.item())
            # Backward pass
            loss.backward()

            train_loss += -loss.item()
            it += 1
            optimizer.step()

            train_loader_tqdm.set_postfix(loss=train_loss / it)
    return cvae, epoch_losses


def predict_one_batch(
    cvae: nn.Module,
    data: torch.Tensor,
    K: int,
    batch_size: int,
    num_classes: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """do a prediction on one batch

    Args:
        cvae (nn.Module): model
        data (torch.Tensor): data
        K (int): number of samples
        batch_size (int): batch size
        num_classes (int): number of classes
        device (str): cuda or cpu

    Returns:
        tuple[torch.Tensor, torch.Tensor]: return the prediction and the loss
    """
    losses = torch.zeros((batch_size, num_classes)).to(device)

    for i in range(num_classes):
        for k in range(K):
            y = torch.zeros(1, num_classes).to(device)
            y[0, i] = 1
            y = y.repeat(1, data.shape[0]).reshape(data.shape[0], -1)

            losses[:, i] += cvae(data, y, 1)

    losses /= K
    res = losses.argmax(axis=1)
    return res, losses


def predict(
    cvae: nn.Module, testloader: DataLoader, device: str, num_classes: int, K: int
) -> tuple[list, list]:
    """do prediction on the testloader

    Args:
        cvae (nn.Module): model
        testloader (DataLoader): test DataLoader
        device (str): cuda or cpu
        num_classes (int): number of classes
        K (int): number of samples

    Returns:
        tuple[list, list]: return the prediction and the target
    """
    cvae.to(device)
    cvae.eval()
    # Lists for metrics
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(testloader)):
            # Skip the last batch
            if batch_idx == len(testloader) - 1:
                continue
            data, target = data.to(device), target.to(device)
            target_one_hot = F.one_hot(target, num_classes=num_classes).to(
                torch.float32
            )
            res, _ = predict_one_batch(cvae, data, K, batch_size, num_classes, device)

            all_preds.extend(res.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    return all_preds, all_targets


def predict_poisoned(
    cvae: nn.Module,
    testloader: DataLoader,
    device: str,
    num_classes: int,
    list_eps: list,
    Q: int,
    K: int,
) -> tuple[list, list, dict]:
    """predict on the testloader with poisoned data (ie after attack)

    Args:
        cvae (nn.Module): model
        testloader (DataLoader): test DataLoader
        device (str): cuda or cpu
        num_classes (int):  number of classes
        list_eps (list): list of epsilon for the attack
        Q (int): number of attack
        K (int): number of samples

    Returns:
        tuple[list, list, dict]: return the prediction, the target and the prediction after attack
    """
    cvae.to(device)
    cvae.eval()
    torch.set_grad_enabled(True)
    # Lists for metrics
    all_preds = []
    all_targets = []
    all_preds_poisoned = defaultdict(list)

    for batch_idx, (data, target) in tqdm(enumerate(testloader)):
        # Skip the last batch
        # if batch_idx == len(testloader) - 1:
        if batch_idx > Q:
            break
        data, target = data.to(device).requires_grad_(True), target.to(device)
        target_one_hot = F.one_hot(target, num_classes=num_classes).to(torch.float32)
        res, _ = predict_one_batch(cvae, data, K, data.shape[0], num_classes, device)
        all_preds.extend(res.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        # plt.imshow(data.cpu().detach().numpy().squeeze(0).reshape((28, 28, 1)))
        # plt.show()
        # print('before attack')
        # print(f'predicted : {num_to_class[res.item()]}, labels : {num_to_class[target_one_hot.argmax().item()]}')
        losses_ = cvae(data, target_one_hot, 1)
        losses_.backward()
        # Access the gradients
        gradient = data.grad
        for eps in list_eps:
            data_poised = data + eps * gradient
            # plt.imshow(data_poised.cpu().detach().numpy().squeeze(0).reshape((28, 28, 1)))
            # plt.show()
            res, _ = predict_one_batch(
                cvae, data_poised, K, data.shape[0], num_classes, device
            )
            # print('After attack')
            # print(f'predicted : {num_to_class[res.item()]}, labels : {num_to_class[target_one_hot.argmax().item()]}')

            all_preds_poisoned[eps].extend(res.cpu().numpy())
    return all_preds, all_targets, all_preds_poisoned


def zoo_attack(
    image: torch.Tensor,
    model: nn.Module,
    target_label: int,
    eps: float,
    device: str,
    max_iterations: int = 200,
    learning_rate: float = 0.01,
    lambda_reg: float = 0.01,
) -> torch.Tensor:
    """
    Perform a ZOO-based adversarial attack.

    Args:
    image (torch.Tensor): The original image on the device (CPU or GPU).
    label (int): The true label of the image.
    model (Callable): The model that returns softmax probabilities.
    max_iterations (int): Maximum number of optimization iterations.
    targeted (bool): Whether the attack is targeted or not.
    target_label (int): The target label for a targeted attack.
    learning_rate (float): The learning rate for the optimization.
    lambda_reg (float): Regularization parameter.

    Returns:
    torch.Tensor: Adversarial image.
    """

    torch.set_grad_enabled(True)
    image = image.requires_grad_(True).float()

    def objective_function(perturbation, model):
        adversarial_img = torch.clamp(image + perturbation, 0, 1)

        _, predictions = predict_one_batch(
            model, adversarial_img, K=10, batch_size=1, num_classes=10, device=device
        )
        loss = -predictions[0, target_label] + lambda_reg * torch.norm(perturbation)
        return loss

    # Initial perturbation
    perturbation = torch.zeros_like(image, requires_grad=True)

    # Run the optimization
    for _ in range(max_iterations):
        loss = objective_function(perturbation, model)
        loss.backward()
        with torch.no_grad():
            perturbation -= learning_rate * perturbation.grad
            perturbation.grad.zero_()

            if torch.norm(perturbation) > eps:
                # Keeping the perturbation small
                break
    adversarial_img = torch.clamp(image + perturbation, 0, 1)
    return adversarial_img


def predict_poisoned_zoo(
    cvae: nn.Module,
    testloader: DataLoader,
    device: str,
    num_classes: int,
    target_label: int,
    list_eps: list,
    Q: int,
    K: int,
) -> tuple[list, list, dict]:
    """predict on the testloader with poisoned data (ie after attack)

    Args:
        cvae (nn.Module): model
        testloader (DataLoader): test DataLoader
        device (str): cuda or cpu
        num_classes (int): number of classes
        target_label (int): target label for the attack
        list_eps (list): list of epsilon for the attack
        Q (int): number of attack
        K (int): number of samples

    Returns:
        tuple[list, list, dict]: return the prediction, the target and the prediction after attack
    """
    cvae.to(device)
    cvae.eval()
    torch.set_grad_enabled(True)
    # Lists for metrics
    all_preds = []
    all_targets = []
    all_preds_poisoned = defaultdict(list)

    for batch_idx, (data, target) in enumerate(testloader):
        print("batch", batch_idx)

        if batch_idx > Q:
            break
        data, target = data.to(device).requires_grad_(True), target.to(device)
        _ = F.one_hot(target, num_classes=num_classes).to(torch.float32)
        res, _ = predict_one_batch(cvae, data, K, data.shape[0], num_classes, device)
        print("batch", res)
        all_preds.extend(res.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

        for eps in list_eps:
            data_poised = zoo_attack(
                data, cvae, target_label=target_label, eps=eps, device=device
            )
            res, _ = predict_one_batch(
                cvae, data_poised, K, data.shape[0], num_classes, device
            )
            print("batch", res)

            all_preds_poisoned[eps].extend(res.cpu().numpy())
    return all_preds, all_targets, all_preds_poisoned


def train_compare_attack_inf(
    trainloader: DataLoader,
    testloader: DataLoader,
    device: str,
    list_eps: list,
    output_path: str,
    dataset: str,
    lr: float,
    num_epochs: int,
    num_classes: int,
    latent_size: int,
    H: int,
    Q: int,
    K: int,
) -> dict:
    """train and compare the different models

    Args:
        trainloader (DataLoader): train DataLoader
        testloader (DataLoader): test DataLoader
        device (str): cuda or cpu
        list_eps (list): list of epsilon for the attack
        output_path (str): path for saving the results
        dataset (str): name of the dataset
        lr (float): learning rate
        num_epochs (int): number of epochs
        num_classes (int): number of classes
        latent_size (int): latent size
        H (int): hidden size
        Q (int): number of attack
        K (int): number of samples

    Returns:
        dict: return the results
    """
    res_by_eps_poisoned = defaultdict(dict)
    for model_name in ["GFZ", "GBZ", "GFY", "GBY", "DBX", "DFX", "DFZ"]:
        print(model_name)
        cvae, loss = train(
            trainloader,
            lr,
            num_epochs,
            num_classes,
            latent_size,
            H,
            name_model=model_name,
            dataset=dataset,
        )
        plot_loss(loss)

        # map_model_results = dict()
        # map_model_results_poisoned = dict()

        all_preds, all_targets, all_preds_poisoned = predict_poisoned(
            cvae, testloader, device, num_classes, list_eps, Q, K
        )
        # normal
        accuracy, precision, recall, f1 = evaluation_vizualization(
            all_preds, all_targets, num_classes
        )
        res_by_eps_poisoned[model_name][0] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        # attack
        for k, v in all_preds_poisoned.items():
            accuracy, precision, recall, f1 = evaluation_vizualization(
                v, all_targets, num_classes
            )
            res_by_eps_poisoned[model_name][k] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        # return res_by_eps_poisoned
    with open(output_path, "w") as f:
        json.dump(res_by_eps_poisoned, f)

    return res_by_eps_poisoned


def train_compare_attack_zoo(
    trainloader: DataLoader,
    testloader: DataLoader,
    device: str,
    target_label: int,
    list_eps: list,
    output_path: str,
    dataset: str,
    lr: float,
    num_epochs: int,
    num_classes: int,
    latent_size: int,
    H: int,
    Q: int,
    K: int,
) -> dict:
    """train and compare the different models

    Args:
        trainloader (DataLoader): train DataLoader
        testloader (DataLoader): test DataLoader
        device (str): cuda or cpu
        target_label (int): target label for the attack
        list_eps (list): list of epsilon for the attack
        output_path (str): path for saving the results
        dataset (str): name of the dataset
        lr (float): learning rate
        num_epochs (int): number of epochs
        num_classes (int): number of classes
        latent_size (int): latent size
        H (int): hidden size
        Q (int): number of attack
        K (int): number of samples

    Returns:
        dict: return the results
    """
    res_by_eps_poisoned = defaultdict(dict)
    for model_name in ["GFZ", "GBZ", "GFY", "GBY", "DBX", "DFX", "DFZ"]:
        print(model_name)
        cvae, loss = train(
            trainloader,
            lr,
            num_epochs,
            num_classes,
            latent_size,
            H,
            name_model=model_name,
            dataset=dataset,
        )
        plot_loss(loss)

        all_preds, all_targets, all_preds_poisoned = predict_poisoned_zoo(
            cvae, testloader, device, num_classes, target_label, list_eps, Q, K
        )
        # normal
        accuracy, precision, recall, f1 = evaluation_vizualization(
            all_preds, all_targets, num_classes
        )
        res_by_eps_poisoned[model_name][0] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        # attack
        for k, v in all_preds_poisoned.items():
            accuracy, precision, recall, f1 = evaluation_vizualization(
                v, all_targets, num_classes
            )
            res_by_eps_poisoned[model_name][k] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

    with open(output_path, "w") as f:
        json.dump(res_by_eps_poisoned, f)

    return res_by_eps_poisoned


def visualisation_attack_zoo(
    cvae: nn.Module,
    testloader: DataLoader,
    target: int,
    eps: list,
    device: str,
    num_classes: int,
) -> None:
    """visualisation of the attack

    Args:
        cvae (nn.Module): model
        testloader (DataLoader): test DataLoader
        target (int):  target label for the attack
        eps (list): list of epsilon for the attack
        device (str): cuda or cpu
        num_classes (int): number of classes

    Returns:
        _type_: None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in range(20):
        i = np.random.randint(128)
        images, labels = next(iter(testloader))

        images_gpu = images.to(device)
        labels = labels.to(device)

        adversarial_image_gpu = zoo_attack(
            images_gpu[i].unsqueeze(0),
            cvae,
            target_label=target,
            eps=eps,
            device=device,
        )

        target_one_hat = (
            torch.nn.functional.one_hot(labels[i], num_classes=num_classes)
            .to(torch.float32)
            .unsqueeze(0)
        )
        # print(target.shape)

        _, original_pred = predict_one_batch(
            cvae,
            images_gpu[i].unsqueeze(dim=0),
            K=10,
            batch_size=1,
            num_classes=num_classes,
            device=device,
        )
        _, adversarial_pred = predict_one_batch(
            cvae,
            adversarial_image_gpu,
            K=10,
            batch_size=1,
            num_classes=num_classes,
            device=device,
        )

        image_cpu = images[i].cpu().detach().numpy().squeeze()
        adversarial_image_cpu = adversarial_image_gpu.cpu().detach().numpy().squeeze()
        original_pred = original_pred.cpu().detach().numpy()
        adversarial_pred = adversarial_pred.cpu().detach().numpy()

        # Plot original and adversarial images
        sns.set(style="whitegrid", context="paper")

        def denormalize(tensor):
            mean = 0.5
            std = 0.5
            denormalized = tensor * std + mean
            return denormalized

        if target == np.argmax(adversarial_pred[0]):
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            # Denormalizing before plotting
            original_image_denormalized = denormalize(image_cpu)
            if len(original_image_denormalized.shape) > 2:
                plt.imshow(original_image_denormalized.transpose((1, 2, 0)))
            else:
                plt.imshow(original_image_denormalized)
            plt.title(
                f"Original Image\nTrue Label: {labels[i].item()}\nPredicted Label: {np.argmax(original_pred[0])}"
            )
            plt.axis("off")

            # Plotting the adversarial image
            plt.subplot(1, 2, 2)
            # Denormalizing before plotting
            adversarial_image_denormalized = denormalize(adversarial_image_cpu)

            if len(original_image_denormalized.shape) > 2:
                plt.imshow(adversarial_image_denormalized.transpose((1, 2, 0)))
            else:
                plt.imshow(adversarial_image_denormalized)
            plt.title(
                f"Adversarial Image\nPredicted Label: {np.argmax(adversarial_pred[0])}"
            )
            plt.axis("off")

            plt.show()
