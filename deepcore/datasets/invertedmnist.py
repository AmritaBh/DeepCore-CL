from torchvision import datasets, transforms
import numpy as np


def x_invert(x):
    return transforms.functional.invert(x)



def IMNIST(data_path, permuted=False, permutation_seed=None):
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.869]
    std = [0.3084]

    # x_invert = lambda x: transforms.functional.invert(x)
    transform = transforms.Compose([transforms.ToTensor(), \
                                    transforms.Lambda(x_invert),\
                                    transforms.Normalize(mean=mean, std=std)])
    if permuted:
        np.random.seed(permutation_seed)
        pixel_permutation = np.random.permutation(28 * 28)
        transform = transforms.Compose(
            [transform, transforms.Lambda(lambda x: x.view(-1, 1)[pixel_permutation].view(1, 28, 28))])

    dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    class_names = [str(c) for c in range(num_classes)]
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test


def permutedIMNIST(data_path, permutation_seed=None):
    return IMNIST(data_path, True, permutation_seed)
