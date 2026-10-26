import argparse
import random
from pathlib import Path

import clip
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10


CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class CifarSubset(Dataset):
    def __init__(self, dataset: CIFAR10, indices: list[int], preprocess) -> None:
        self.dataset = dataset
        self.indices = indices
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.dataset[self.indices[index]]
        return self.preprocess(image), label


def select_balanced_indices(dataset: CIFAR10, images_per_class: int) -> list[int]:
    selected: list[int] = []
    counts = [0] * len(CLASS_NAMES)
    for index, label in enumerate(dataset.targets):
        if counts[label] >= images_per_class:
            continue
        selected.append(index)
        counts[label] += 1
        if all(count == images_per_class for count in counts):
            break
    return selected


def save_dataset_images(dataset: CIFAR10, indices: list[int], output_dir: Path) -> None:
    for index in indices:
        image, label = dataset[index]
        class_dir = output_dir / CLASS_NAMES[label]
        class_dir.mkdir(parents=True, exist_ok=True)
        image.save(class_dir / f"{index:05d}.png")


def generate_synthetic_images(output_dir: Path, images_per_class: int) -> list[tuple[Path, int]]:
    random.seed(42)
    generated = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(images_per_class):
            image = Image.new("RGB", (224, 224), (20 + label * 20, 35, 70))
            draw = ImageDraw.Draw(image)
            color = (220, 80 + label * 12, 60 + index * 10)
            margin = 30 + random.randint(-8, 8)
            if label % 3 == 0:
                draw.ellipse((margin, margin, 224 - margin, 224 - margin), fill=color)
            elif label % 3 == 1:
                draw.rectangle((margin, margin, 224 - margin, 224 - margin), fill=color)
            else:
                draw.polygon(
                    ((112, margin), (224 - margin, 224 - margin), (margin, 224 - margin)),
                    fill=color,
                )
            path = class_dir / f"{index:05d}.png"
            image.save(path)
            generated.append((path, label))
    return generated


class ImageFileDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]], preprocess) -> None:
        self.items = items
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        return self.preprocess(Image.open(path).convert("RGB")), label


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path(args.data_root)
    image_root = data_root / "images"
    data_root.mkdir(parents=True, exist_ok=True)

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    if args.source == "synthetic":
        items = generate_synthetic_images(image_root, args.images_per_class)
    else:
        cifar = CIFAR10(root=data_root / "cifar10", train=True, download=True)
        indices = select_balanced_indices(cifar, args.images_per_class)
        save_dataset_images(cifar, indices, image_root)
        items = [(image_root / CLASS_NAMES[cifar.targets[index]] / f"{index:05d}.png", cifar.targets[index]) for index in indices]

    dataset = ImageFileDataset(items, preprocess) if args.source == "synthetic" else CifarSubset(cifar, indices, preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    classifier = nn.Linear(clip_model.visual.output_dim, len(CLASS_NAMES)).to(device)

    # Fine-tune the image encoder and learn a task-specific classification head.
    optimizer = torch.optim.AdamW(
        list(clip_model.visual.parameters()) + list(classifier.parameters()),
        lr=args.learning_rate,
    )
    loss_fn = nn.CrossEntropyLoss()
    clip_model.train()
    classifier.train()

    for epoch in range(args.epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings = clip_model.encode_image(images).float()
            logits = classifier(embeddings)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"loss={total_loss / total:.4f}, accuracy={correct / total:.3f}"
        )

    checkpoint = {
        "class_names": CLASS_NAMES,
        "classifier": classifier.state_dict(),
        "visual": clip_model.visual.state_dict(),
    }
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved {len(dataset)} images to {image_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on a 100-image CIFAR-10 subset.")
    parser.add_argument("--data-root", default="datasets/cifar10_100")
    parser.add_argument("--source", choices=("synthetic", "cifar10"), default="synthetic")
    parser.add_argument("--checkpoint", default="models/clip-cifar10-100.pt")
    parser.add_argument("--images-per-class", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
