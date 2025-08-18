from data_loader import load_dataset

train_loader, val_loader, test_loader, subtype_map = load_dataset(
    root_dir="dataset",
    img_size=224,
    batch_size=32
)

print("Subtype Mapping:", subtype_map)

for imgs, binary, subtype in train_loader:
    print(imgs.shape, binary, subtype)
    break