import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from cellpose import models, io, train, core
from cellpose.train import train_seg
from sklearn.model_selection import train_test_split


def read_txt(txt_file):
    with open(txt_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def split_data(image_list, ratio):
    return train_test_split(image_list, train_size=ratio, random_state=42)


def load_images_and_masks(paths):
    imgs, masks = [], []
    for line in paths:
        img_path, mask_path = line.split(',')
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        imgs.append(io.imread(img_path))
        masks.append(io.imread(mask_path))
    return imgs, masks


def save_loss_plot(log_path, save_dir):
    with open(log_path, 'r') as f:
        logs = json.load(f)
    epochs = [entry['epoch'] for entry in logs]
    train_loss = [entry['train_loss'] for entry in logs]
    val_loss = [entry['val_loss'] for entry in logs]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o', markersize=3)
    plt.plot(epochs, val_loss, label='Val Loss', marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()


def train_model(args):
    print("Starting Cellpose fine-tuning...")

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("finetuned_models", f"cellpose_{args.type}_{current_time}")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train_log.json")

    image_paths = read_txt(args.txt_file)
    train_paths, val_paths = split_data(image_paths, args.ratio)
    train_imgs, train_masks = load_images_and_masks(train_paths)
    val_imgs, val_masks = load_images_and_masks(val_paths)

    use_gpu = core.use_gpu()
    print(f">>>> using {'GPU' if use_gpu else 'CPU'}")

    pretrained = None if args.pretrained_model == "scratch" else args.pretrained_model
    model = models.CellposeModel(gpu=use_gpu, model_type=pretrained)
    channels = [0, 0] if args.type == 'ss' else [1, 3]

    n_epochs = args.nb_epoch
    patience = 30
    best_val_loss = float('inf')
    no_improve_count = 0
    batch_size = args.batch_size
    logs = []
    best_model_path = os.path.join(output_dir, "best_model.pth")

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")

        train_loss, _ = train_seg(
            model.net, train_imgs, train_masks,
            test_data=None, test_labels=None,
            channels=channels, save_path=None,
            n_epochs=1, learning_rate=0.1,
            SGD=True, weight_decay=1e-4,
            batch_size=batch_size, rescale=False, nimg_per_epoch=8,
        )

        val_loss, _ = train_seg(
            model.net, val_imgs, val_masks,
            test_data=None, test_labels=None,
            channels=channels, save_path=None,
            n_epochs=1, learning_rate=0.0,
            SGD=False, weight_decay=0,
            batch_size=batch_size, rescale=False, nimg_per_epoch=8,
        )

        logs.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss)
        })
        with open(log_path, "w") as f:
            json.dump(logs, f, indent=2)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            no_improve_count = 0
            model.net.save_model(path=best_model_path)
            print(">>> Best model saved.")
        else:
            no_improve_count += 1
            print(f">>> No improvement for {no_improve_count} epochs.")

        if no_improve_count >= patience:
            print(">>> Early stopping.")
            break

    save_loss_plot(log_path, output_dir)
    print(f"\nTraining finished. Best model saved to: {best_model_path}")


def train(args):
    train_model(args)