import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import config

def get_data_loaders():
    print("Loading dataset...")

    data = torch.load(config.PROCESSED_FILE)
    X = data["features"]
    y = data["labels"].long()
    input_dim = data["input_dim"]

    print("Unique labels:", torch.unique(y))
    
    dataset = TensorDataset(X, y)

    train_size = int(len(dataset) * config.TRAIN_SPLIT)
    val_size = int(len(dataset) * config.VAL_SPLIT)
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, input_dim