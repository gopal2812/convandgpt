import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary

train_losses = []
test_losses = []
train_acc = []
test_acc = []
test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

def get_device():
    # if torch.cuda.is_available():
    #     device = "cuda"
    # else:
    #     device = "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def plot_sample_images(train_dataloader):
    batch_data, batch_label = next(iter(train_dataloader)) 

    fig = plt.figure()

    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

def get_model_summary(model, sample_input_sz=(1,28,28)):
    # default is for MNIST inputs, should override for other datasets
    #use_cuda = torch.cuda.is_available()
    device = get_device() #torch.device("cuda" if use_cuda else "cpu")
    model_instance = model().to(device)
    summary(model_instance,input_size=sample_input_sz )

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer):
  train_losses = []
  #test_losses = []
  train_acc = []
  #test_acc = []


  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = F.nll_loss(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))

  return train_acc, train_losses

def test(model, device, test_loader):
    #train_losses = []
    test_losses = []
    #train_acc = []
    test_acc = []
    test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}


    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return test_acc, test_losses



def get_default_mnist_transforms():
   train_transforms = transforms.Compose([
      transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
      transforms.Resize((28, 28)),
      transforms.RandomRotation((-15., 15.), fill=0),
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,)),
    ])

 # Test data transformations
   test_transforms = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
      ])
   return train_transforms, test_transforms


def get_mnist_dataset():
   train_transforms, test_transforms = get_default_mnist_transforms()
   train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
   test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

   return train_data, test_data

def get_data_loaders(train_data, test_data, batch_size):
   kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
   train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
   test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

   return train_loader, test_loader
