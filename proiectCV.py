import os
import time
import zipfile
from typing import Iterable, Tuple

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_DIR = "./processed_data"
ARCHIVE_NAME = "detector emotii.zip"
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0005
VAL_SPLIT = 0.2
MODEL_PATH = "./model.pth"
RANDOM_SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Folosește device-ul:", device)

class CNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

def ensure_dataset() -> None:
    if os.path.isdir(DATA_DIR) and any(os.scandir(DATA_DIR)):
        return

    zip_path = os.path.join(os.getcwd(), ARCHIVE_NAME)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Arhiva cu date nu a fost găsită la {zip_path}. Pune fișierul lângă script."
        )

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(os.getcwd())
    print(f"Dataset extras din {zip_path}!")

    if not (os.path.isdir(DATA_DIR) and any(os.scandir(DATA_DIR))):
        raise RuntimeError(
            f"După extragere nu am găsit folderul {DATA_DIR}. Verifică structura arhivei."
        )


def _split_indices(total: int) -> Tuple[Iterable[int], Iterable[int]]:
    if total < 2:
        raise RuntimeError("Este nevoie de cel puțin două imagini în set.")

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    permutation = torch.randperm(total, generator=generator).tolist()

    val_count = max(1, int(total * VAL_SPLIT))
    if val_count >= total:
        val_count = max(1, total - 1)

    val_indices = permutation[:val_count]
    train_indices = permutation[val_count:]

    if not train_indices:
        raise RuntimeError(
            "Setul de antrenare este gol. Adaugă mai multe imagini sau micșorează VAL_SPLIT."
        )

    return train_indices, val_indices


def prepare_dataloaders() -> Tuple[DataLoader, DataLoader, list[str]]:
    ensure_dataset()

    base_dataset = datasets.ImageFolder(DATA_DIR)
    class_names = base_dataset.classes
    total_samples = len(base_dataset)

    train_indices, val_indices = _split_indices(total_samples)

    train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    num_workers = 0 if device.type == "cpu" else 2
    pin_memory = device.type == "cuda"
    #datal
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, list(class_names)

def train() -> None:
    train_loader, val_loader, class_names = prepare_dataloaders()

    model = CNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    print(
        f"Încep antrenarea pe {len(train_loader.dataset)} imagini (validare: {len(val_loader.dataset)})."
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        start_time = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        avg_loss = running_loss / total
        avg_acc = running_correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += images.size(0)

        avg_val_loss = val_loss / val_total
        avg_val_acc = val_correct / val_total
        elapsed = time.time() - start_time

        print(
            f"EPOCH {epoch}/{EPOCHS}  train_loss={avg_loss:.3f} train_acc={avg_acc:.3f} "
            f"val_loss={avg_val_loss:.3f} val_acc={avg_val_acc:.3f} time={elapsed:.1f}s"
        )

        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "img_size": IMG_SIZE,
                },
                MODEL_PATH,
            )
            print(
                f"Model salvat cu acuratețe de validare {best_acc:.3f} în {MODEL_PATH}."
            )

    print("Training completed!")

# ---------------- INFERENCE FUNCTION ----------------
def inference() -> None:
    ensure_dataset()

    if not os.path.exists(MODEL_PATH):
        print(f"Modelul nu există la {MODEL_PATH}.")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names: list[str] = checkpoint["class_names"]
    img_size: int = checkpoint["img_size"]

    model = CNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Nu am putut încărca clasificatorul Haar pentru față.")

    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Webcam error")
        return

    try:
        while True:
            ret, frame = webcam.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )

            if len(faces) == 0:
                h, w, _ = frame.shape
                size = min(h, w)
                start_y = (h - size) // 2
                start_x = (w - size) // 2
                face_roi = frame[start_y:start_y + size, start_x:start_x + size]
            else:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_roi = frame[y:y + h, x:x + w]

            rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            resize_img = Image.fromarray(rgb).resize((img_size, img_size))
            gray_img = resize_img.convert("L").convert("RGB")
            tensor = transforms.functional.to_tensor(gray_img)
            tensor = transforms.functional.normalize(
                tensor,
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            )
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(tensor)
                probs = torch.nn.functional.softmax(out, dim=1)
                top_prob, prob_idx = torch.max(probs, dim=1)
                label = class_names[prob_idx.item()]
                conf = top_prob.item()

            if conf < 0.5:
                ordered = torch.topk(probs.cpu().squeeze(), k=min(3, len(class_names)))
                print("Confidență mică. Top scoruri:")
                for idx, score in zip(ordered.indices.tolist(), ordered.values.tolist()):
                    print(f"    {class_names[idx]}: {score:.2f}")

            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Emotion Detector (q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        webcam.release()
        cv2.destroyAllWindows()

# ---------------- RUN ----------------
if __name__ == "__main__":
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Nu am găsit modelul la {MODEL_PATH}. Pornesc antrenarea...")
            train()
        else:
            print(f"Modelul există deja la {MODEL_PATH}. Sar peste antrenare.")

        inference()
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"Eroare: {exc}")
