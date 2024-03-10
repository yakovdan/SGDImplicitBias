import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def load_dataset_to_gpu(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    inputs_np_array = dataset.data
    labels_np_array = np.array(dataset.targets).reshape((-1, 1))
    inputs_gpu_tensor = torch.tensor(inputs_np_array, device="cuda").permute([0, 3, 1, 2]).contiguous() / 255
    transforms.functional.normalize(inputs_gpu_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
    labels_gpu_tensor = torch.tensor(labels_np_array, device="cuda")
    return inputs_gpu_tensor, labels_gpu_tensor


def imshow(img: torch.Tensor) -> None:
    return
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    return


def calculate_conv_weights_rank(weights_tensor: torch.Tensor, tol=0.001) -> float:
    weights_tensor = weights_tensor.reshape((weights_tensor.shape[0], -1))
    singular_values = torch.linalg.svdvals(weights_tensor)
    singular_values[1:] /= singular_values[0]  # equivalent to division by matrix 2-norm
    singular_values[0] = 1
    result = (singular_values > tol).sum().item()
    return result


def calculate_average_conv_rank(model: nn.Module, tol=0.001) -> float:
    parameter_count = len(list(model.parameters()))
    named_params = list(model.named_parameters())
    weights = []
    for i in range(parameter_count):
        if named_params[i][1].ndim == 2:
            weights.append(named_params[i][1])
    weight_ranks = []
    for weight in weights:
        weight_ranks.append(calculate_conv_weights_rank(weight, tol))
    avg_rank: float = sum(weight_ranks) / len(weight_ranks)
    return avg_rank


EPOCHS: int = 10
LR: float = 0.1
BATCH_SIZE: int = 32
PATH: str = './cifar10_resnet_SGD.pth'
WEIGHT_DECAY: float = 8e-4
MOMENTUM = 0.9

class MLP_BN_L_H(nn.Module):
    def __init__(self, L: int, H: int, input_dim: int = 3 * 32**2, bn: bool = True, num_classes=10):
        super().__init__()
        layers = [nn.Sequential(nn.Linear(in_features=input_dim, out_features=H),
                                nn.BatchNorm1d(num_features=H),
                                nn.ReLU())]
        for i in range(1, L):
            layers.append(nn.Sequential(nn.Linear(in_features=H, out_features=H),
                                        nn.BatchNorm1d(num_features=H),
                                        nn.ReLU()))

        layers.append(nn.Linear(H, num_classes))
        self.layers = nn.Sequential(*layers)

        # self.fc = nn.Linear(in_features=input_dim, out_features=H)
        # self.bn = nn.BatchNorm1d(num_features=H)
        # self.act = nn.ReLU()

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.layers(x)



if __name__ == "__main__":
    print(torch.__version__)
    torch.cuda.is_available()

    trained_resnet: nn.Module = resnet18(weights=ResNet18_Weights.DEFAULT)
    untrained_resnet: nn.Module = resnet18(weights=None, num_classes=10).cuda()

    #model = untrained_resnet
    mlp_bn_10_100 = MLP_BN_L_H(L=10, H=100)
    model = mlp_bn_10_100.cuda()

    train_transform: torchvision.transforms.Compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.RandomCrop(size=28),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(15),
         transforms.Resize(size=32),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_transform: torchvision.transforms.Compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set: Dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                      download=True, transform=train_transform)

    train_loader: DataLoader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                                           shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
                                                           )
    test_set: Dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                     download=True, transform=test_transform)
    test_loader: DataLoader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                                          shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    classes: tuple[str, ...] = ('plane', 'car', 'bird', 'cat',
                                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # get some random training images
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    grid_images = torchvision.utils.make_grid(images)
    # show images
    imshow(grid_images)
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 100, 200], gamma=0.1)
    inputs_gpu_train, labels_gpu_train = load_dataset_to_gpu(train_set)
    inputs_gpu_test, labels_gpu_test = load_dataset_to_gpu(test_set)

    # train
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0

        for i, data in tqdm(enumerate(train_loader), total=50000 // BATCH_SIZE):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].cuda(), data[1].cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        scheduler.step()
        model.eval()
        correct = 0
        total = 0

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0].cuda(), data[1].cuda()
                # calculate outputs by running images through the network
                outputs = model(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_rank = calculate_average_conv_rank(model, tol=0.001)
            print(f"avg rank: {avg_rank}")
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    print('Finished Training')
    torch.save(model.state_dict(), PATH)

    #test
    model.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].cuda(), data[1].cuda()
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # prepare to count predictions for each class

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].cuda(), data[1].cuda()
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
