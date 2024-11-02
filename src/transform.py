from torchvision.transforms import Compose, Normalize, ToTensor


def get_transform():
    return Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def get_test_transform():
    return Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
