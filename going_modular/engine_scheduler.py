import torch

import torch.optim as optim
from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def adjust_learning_rate(optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler.LambdaLR, epoch: int, max_lr_epoch: int) -> None:
    if epoch <= max_lr_epoch:
        lr_scheduler.step()  # Adjust learning rate according to the scheduler
    else:
        for _ in range(epoch - max_lr_epoch):  # Keep learning rate unchanged
            lr_scheduler.step()


def caculate_loss_weight(epoch: int) -> Tuple[float, float]:
    if epoch < 10:
        return 0.9, 0.1
    if epoch < 20:
        return 0.5, 0.5
    if epoch < 30:
        return 0.1, 0.9
    else:
        return 0.02, 0.98


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn1: torch.nn.Module,
               loss_fn2: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               loss_weights: Tuple[float, float],
               scheduler: torch.optim.lr_scheduler.LambdaLR,
               device: torch.device) -> Tuple[float, float]:
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc1, train_acc2 = 0, 0, 0
    weight1, weight2 = loss_weights

    # Loop through data loader data batches
    for batch, (X, y1, y2) in enumerate(dataloader):
        # Send data to target device
        X, y1, y2 = X.to(device), y1.to(device), y2.to(device)

        # 1. Forward pass
        y1_pred, y2_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss1 = loss_fn1(y1_pred, y1)
        loss2 = loss_fn2(y2_pred, y2)
        loss = ((loss1 * weight1) + (loss2 * weight2)) / 2
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y1_pred_class = torch.argmax(y1_pred, dim=1)
        y2_pred_class = torch.argmax(y2_pred, dim=1)
        # y_pred_class = torch.argmax(y_pred, dim=1)
        train_acc1 += (y1_pred_class == y1).sum().item() / len(y1_pred)
        train_acc2 += (y2_pred_class == y2).sum().item() / len(y2_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc1 = train_acc1 / len(dataloader)
    train_acc2 = train_acc2 / len(dataloader)

    return train_loss, train_acc1, train_acc2


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn1: torch.nn.Module,
              loss_fn2: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc1, test_acc2 = 0, 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y1, y2) in enumerate(dataloader):
            # Send data to target device
            X, y1, y2 = X.to(device), y1.to(device), y2.to(device)

            # 1. Forward pass
            test1_pred_logits, test2_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss1 = loss_fn1(test1_pred_logits, y1)
            loss2 = loss_fn2(test2_pred_logits, y2)
            loss = (loss1 + loss2) / 2
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test1_pred_labels = test1_pred_logits.argmax(dim=1)
            test2_pred_labels = test2_pred_logits.argmax(dim=1)
            test_acc1 += ((test1_pred_labels == y1).sum().item() / len(test1_pred_labels))
            test_acc2 += ((test2_pred_labels == y2).sum().item() / len(test2_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc1 = test_acc1 / len(dataloader)
    test_acc2 = test_acc2 / len(dataloader)
    return test_loss, test_acc1, test_acc2


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn1: torch.nn.Module,
          loss_fn2: torch.nn.Module,
          scheduler: torch.optim.lr_scheduler.LambdaLR,
          epochs: int,
          device: torch.device,
          max_lr_epoch: int) -> Dict[str, List]:
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc1": [],
               "train_acc2": [],
               "test_loss": [],
               "test_acc1": [],
               "test_acc2": []
               }

    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        loss_weights = caculate_loss_weight(epochs)
        train_loss, train_acc1, train_acc2 = train_step(model=model,
                                                        dataloader=train_dataloader,
                                                        loss_fn1=loss_fn1,
                                                        loss_fn2=loss_fn2,
                                                        optimizer=optimizer,
                                                        device=device,
                                                        loss_weights=loss_weights)

        # Adjust learning rate
        adjust_learning_rate(optimizer=optimizer, lr_scheduler=scheduler, epoch=epoch, max_lr_epoch=max_lr_epoch)

        test_loss, test_acc1, test_acc2 = test_step(model=model,
                                                    dataloader=test_dataloader,
                                                    loss_fn1=loss_fn1,
                                                    loss_fn2=loss_fn2,
                                                    device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc1: {train_acc1:.4f} | "
            f"train_acc2: {train_acc2:.4f} ||| "

            f"test_loss: {test_loss:.4f} | "
            f"test_acc1: {test_acc1:.4f} | "
            f"test_acc2: {test_acc2:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc1"].append(train_acc1)
        results["train_acc2"].append(train_acc2)
        results["test_loss"].append(test_loss)
        results["test_acc1"].append(test_acc1)
        results["test_acc2"].append(test_acc2)

    # Return the filled results at the end of the epochs
    return results
