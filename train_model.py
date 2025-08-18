import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

if __name__ == "__main__":
    data_dir = "dataset_split"
    batch_size = 32
    num_epochs = 25
    learning_rate = 0.001
    num_classes = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225]),
        ]),
    }

    print("Loading datasets...")
    image_datasets = {x: datasets.ImageFolder(root=f"{data_dir}/{x}", 
                                             transform=data_transforms[x]) for x in ["train", "val"]}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=="train"), num_workers=4) for x in ["train", "val"]}
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    print(f"Classes: {class_names}")
    print(f"Train size: {dataset_sizes['train']}")
    print(f"Val size: {dataset_sizes['val']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze classifier layers
    for param in model.classifier.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 10)

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                dataloader = dataloaders[phase]
                loop = tqdm(dataloader, leave=False)

                for inputs, labels in loop:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    loop.set_description(f"{phase} Epoch [{epoch+1}/{num_epochs}]")
                    loop.set_postfix(loss=loss.item())

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            scheduler.step()

            # Unfreeze all layers for fine-tuning after 10 epochs
            if epoch == 9:
                print("Unfreezing all layers for fine-tuning...")
                for param in model.parameters():
                    param.requires_grad = True
                optimizer = optim.Adam(model.parameters(), lr=learning_rate / 10)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - 10)

        print(f"\nBest val Acc: {best_acc:.4f}")
        model.load_state_dict(best_model_wts)
        return model

    trained_model = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)

    torch.save(trained_model.state_dict(), "model3.pth")
    print("Training complete. model saved")
