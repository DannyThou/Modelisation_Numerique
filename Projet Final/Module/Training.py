
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import copy
import os

def train(dataloader, model, loss_fn, optimizer, device):
    n_tot = len(dataloader.dataset)
    model.train()
    loss_tot = 0

    for i_batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X.unsqueeze(1)).squeeze(1)
        loss = loss_fn(pred, y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i_batch % 10 == 0:
            current = i_batch * X.size(0)
            print(f"Loss: {loss.item():.6f} [{current}/{n_tot}]")

        loss_tot += loss.item()

    return loss_tot / len(dataloader)


def evaluate(dataloader, model, loss_fn, device, threshold):
    model.eval()
    val_tot = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X, y in dataloader:
            
            X = X.to(device)
            y = y.to(device)

            pred = model(X.unsqueeze(1)).squeeze(1)

            val_tot += loss_fn(pred, y.float()).item()

            probs = torch.sigmoid(pred)
            preds = (probs >= threshold).int()

            y_true.extend(y.cpu().numpy().ravel())
            y_pred.extend(preds.cpu().numpy().ravel())

    return val_tot / len(dataloader), np.array(y_true), np.array(y_pred)


def train_and_evaluate(train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs,device,scheduler, early_stopper,threshold):
    mod_path = "Modèles"
    try:
        os.makedirs(mod_path, exist_ok=True)  #Créer le dossier s'il n'existe pas déjà
    except PermissionError:
        print(f"Permission refusée pour créer le dossier '{mod_path}'")
    except Exception as e:
        print(f"Erreur lors de la création du dossier '{mod_path}': {e}")
    train_losses = []
    val_losses = []
    accuracy = []
    recall = []
    f1_scores = []
    precision = []

    best_val_loss = float('inf')
    best_f1_score = 0

    best_epoch = 0
    best_epoch_f1 = 0

    stopped_early = False

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        train_loss = train(train_dataloader, model, loss_fn, optimizer, device)
        val_loss, y_true, y_pred = evaluate(val_dataloader, model, loss_fn, device,threshold)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        cm = confusion_matrix(y_true,y_pred, labels=[0, 1])

        TN=cm[0,0]
        FP=cm[0,1]
        FN=cm[1,0]
        TP=cm[1,1]

        acc = (TP+TN)/(TP+TN+FP+FN)
        prec = TP/(TP+FP) if (TP+FP) > 0 else 0
        rec = TP/(TP+FN) if (TP+FN) > 0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)
        f1_scores.append(f1)

        if val_loss < best_val_loss:
            best_epoch = t
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(), 
                    "Confusion Matrix": cm.tolist()
                    }, 
                    r"Modèles\best_model.pth"
                    )

        if f1 > best_f1_score:
            best_f1_score = f1
            best_epoch_f1 = t
            torch.save(
                {
                    "model_state_dict": model.state_dict(), 
                    "Confusion Matrix": cm.tolist()
                    }, 
                    r"Modèles\best_model_f1.pth"
                    )
        scheduler.step(val_loss)

        clear_output(wait=True)

        plot_losses(train_losses, val_losses)

        plot_metrics(accuracy, precision, recall, f1_scores)
        
        if early_stopper(val_loss):
            stopped_early = True
            break



    best_model_val = copy.deepcopy(model)
    best_model_f1 = copy.deepcopy(model)

    checkpoint = torch.load(r"Modèles\best_model.pth", map_location=device)
    best_model_val.load_state_dict(checkpoint["model_state_dict"])
    cm = np.array(checkpoint["Confusion Matrix"])
    print(f"Meilleur eval loss: {best_val_loss:.6f}")

    checkpoint_f1 = torch.load(r"Modèles\best_model_f1.pth", map_location=device)
    best_model_f1.load_state_dict(checkpoint_f1["model_state_dict"])
    cm_f1 = np.array(checkpoint_f1["Confusion Matrix"])
    print(f"Meilleur F1-Score: {best_f1_score:.6f}")

    clear_output(wait=True)
    if stopped_early:
        print(f"Early stopping à l'époque {t+1}")
    else:
        print(f"Entraînement terminé après {epochs} époques")

    print(f"Époque de sauvegarde du meilleur modèle Eval Loss: {best_epoch+1}")
    print(f"Époque de sauvegarde du meilleur modèle F1: {best_epoch_f1+1}")

    plot_losses(train_losses, val_losses, best_epoch=best_epoch, best_epoch_f1=best_epoch_f1)
    plot_metrics(accuracy, precision, recall, f1_scores, best_epoch=best_epoch, best_epoch_f1=best_epoch_f1)

    os.remove(r"Modèles\best_model_f1.pth")
    os.remove(r"Modèles\best_model.pth")

    return train_losses, val_losses, cm, best_val_loss, cm_f1, best_f1_score, best_model_val, best_model_f1


def plot_losses(train_losses, val_losses,best_epoch=None,best_epoch_f1=None):
    plt.figure(1)
    plt.plot(np.arange(1, len(train_losses)+1), train_losses, label="Entrainement")
    plt.plot(np.arange(1, len(val_losses)+1), val_losses, label="Validation")
    if best_epoch is not None:
        plt.scatter(best_epoch+1, val_losses[best_epoch], color='red', label="Meilleur modèle Eval Loss", marker='s')
        plt.scatter(best_epoch+1, train_losses[best_epoch], color='red', marker='s')
    if best_epoch_f1 is not None:
        plt.scatter(best_epoch_f1+1, val_losses[best_epoch_f1], color='green', label="Meilleur modèle F1")
        plt.scatter(best_epoch_f1+1, train_losses[best_epoch_f1], color='green')
    plt.title("BCE Loss au fil des époques")
    plt.xlabel("Époques")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.show()

def plot_metrics(accuracy, precision, recall, f1_scores,best_epoch=None,best_epoch_f1=None):
    plt.figure(2,figsize=(12, 8))

    plt.subplot(2,2,1)
    plt.plot(np.arange(1, len(accuracy)+1), accuracy, label="Exactitude")
    if best_epoch is not None:
        plt.scatter(best_epoch+1, accuracy[best_epoch], color='red', label="Meilleur modèle Eval Loss",marker='s')
    if best_epoch_f1 is not None:
        plt.scatter(best_epoch_f1+1, accuracy[best_epoch_f1], color='green', label="Meilleur modèle F1")
    plt.xlabel("Époques")
    plt.ylabel("Exactitude")
    plt.title("Exactitude au fil des époques")
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot(np.arange(1, len(precision)+1), precision, label="Précision")
    if best_epoch is not None:
        plt.scatter(best_epoch+1, precision[best_epoch], color='red', label="Meilleur modèle Eval Loss", marker='s')
    if best_epoch_f1 is not None:
        plt.scatter(best_epoch_f1+1, precision[best_epoch_f1], color='green', label="Meilleur modèle F1")
    plt.xlabel("Époques")
    plt.ylabel("Précision")
    plt.title("Précision au fil des époques")
    plt.legend()

    plt.subplot(2,2,3)
    plt.plot(np.arange(1, len(recall)+1), recall, label="Rappel")
    if best_epoch is not None:
        plt.scatter(best_epoch+1, recall[best_epoch], color='red', label="Meilleur modèle Eval Loss",marker='s')
    if best_epoch_f1 is not None:
        plt.scatter(best_epoch_f1+1, recall[best_epoch_f1], color='green', label="Meilleur modèle F1")
    plt.xlabel("Époques")
    plt.ylabel("Rappel")
    plt.title("Rappel au fil des époques")
    plt.legend()

    plt.subplot(2,2,4)
    plt.plot(np.arange(1, len(f1_scores)+1), f1_scores, label="F1-Score")
    if best_epoch is not None:
        plt.scatter(best_epoch+1, f1_scores[best_epoch], color='red', label="Meilleur modèle Eval Loss", marker='s')
    if best_epoch_f1 is not None:
        plt.scatter(best_epoch_f1+1, f1_scores[best_epoch_f1], color='green', label="Meilleur modèle F1")

    plt.xlabel("Époques")
    plt.ylabel("F1-Score")
    plt.title("F1-Score au fil des époques")
    plt.legend()

    plt.tight_layout()

    plt.show()