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


def calculate_conv_weights_rank(np_weights: np.ndarray, tol=0.001) -> float:
    np_weights = np_weights.reshape((np_weights.shape[0], -1))
    m = np_weights / np.linalg.norm(np_weights, ord=2)
    u, s, vh = np.linalg.svd(m)
    result = (np.abs(s) > tol).sum()
    return result


def calculate_average_conv_rank(model: nn.Module, tol=0.001) -> float:
    parameter_count = len(list(model.parameters()))
    named_params = list(model.named_parameters())
    conv_weight_indices = []
    for i in range(parameter_count):
        if 'conv' in named_params[i][0] and 'bn' in named_params[i + 1][0] and named_params[i][1].shape[0] == \
                named_params[i + 1][1].shape[0]:
            conv_weight_indices.append(i)
    print(len(conv_weight_indices))
    fused_weights: list[np.ndarray] = []
    for i in conv_weight_indices:
        conv_weight = np.copy(named_params[i][1].detach().cpu().numpy())
        bn_weight = np.copy(named_params[i+1][1].detach().cpu().numpy())[:, None, None, None]
        fused_weight: np.ndarray = conv_weight * bn_weight
        fused_weights.append(fused_weight)
    fused_weight_ranks = []
    for fused_weight in fused_weights:
        fused_weight_ranks.append(calculate_conv_weights_rank(fused_weight, tol))
    avg_rank: float = sum(fused_weight_ranks) / len(fused_weight_ranks)
    return avg_rank


EPOCHS: int = 500
LR: float = 5e-3
BATCH_SIZE: int = 4
PATH: str = './cifar10_resnet_SGD.pth'
WEIGHT_DECAY: float = 6e-3

if __name__ == "__main__":
    print(torch.__version__)
    torch.cuda.is_available()

    trained_resnet: nn.Module = resnet18(weights=ResNet18_Weights.DEFAULT)
    untrained_resnet: nn.Module = resnet18(weights=None, num_classes=10).cuda()

    transform: torchvision.transforms.Compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set: Dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                      download=True, transform=transform)

    train_loader: DataLoader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                                           shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
                                                           )
    test_set: Dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                     download=True, transform=transform)
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
    optimizer = optim.SGD(untrained_resnet.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 100, 200], gamma=0.1)
    inputs_gpu_train, labels_gpu_train = load_dataset_to_gpu(train_set)
    inputs_gpu_test, labels_gpu_test = load_dataset_to_gpu(test_set)

    # train
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        untrained_resnet.train()
        input_indices = torch.randperm(inputs_gpu_train.shape[0], device="cuda").reshape((-1, BATCH_SIZE))
        running_loss = 0.0

        for i in tqdm(torch.arange(0, input_indices.shape[0])):
            inputs, labels = inputs_gpu_train[input_indices[i]], labels_gpu_train[input_indices[i]]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = untrained_resnet(inputs)
            loss = criterion(outputs, labels.flatten())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        scheduler.step()
        untrained_resnet.eval()
        correct = 0
        total = 0

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            # calculate outputs by running images through the network
            outputs = untrained_resnet(inputs_gpu_test)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels_gpu_test.shape[0]
            correct += (predicted.reshape((-1, 1)) == labels_gpu_test).sum().item()

            avg_rank = calculate_average_conv_rank(untrained_resnet, tol=0.001)
            print(f"avg rank: {avg_rank}")
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    print('Finished Training')
    torch.save(untrained_resnet.state_dict(), PATH)

    #test:

    # dataiter = iter(test_loader)
    # images, labels = next(dataiter)
    #
    # # print images
    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    untrained_resnet.eval()

    # outputs = untrained_resnet(images.cuda())
    # _, predicted = torch.max(outputs, 1)
    #
    # print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
    #                               for j in range(4)))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        # calculate outputs by running images through the network
        outputs = untrained_resnet(inputs_gpu_test)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels_gpu_test.shape[0]
        correct += (predicted.reshape((-1, 1)) == labels_gpu_test).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        outputs = untrained_resnet(inputs_gpu_test)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.reshape((-1, 1))
        # collect the correct predictions for each class
        for i in range(predictions.shape[0]):
            if labels_gpu_test[i] == predictions[i]:
                correct_pred[classes[labels_gpu_test[i].item()]] += 1
            total_pred[classes[labels_gpu_test[i].item()]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')