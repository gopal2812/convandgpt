from torchvision import transforms

# Train data transformations
train_transforms = transforms.Compose(
    [
        transforms.RandomApply([transforms.RandAugment()], p=0.25),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)

# Test data transformations
test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)
