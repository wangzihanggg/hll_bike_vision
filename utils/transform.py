from torchvision import transforms

train_transform = transforms.Compose([
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
])
