import os
import random
import pickle
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import datasets, transforms

class CIFAR10Dataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.cifar_dataset = datasets.CIFAR10(root=root_dir, train=train, download=True, transform=transform)
        self.classes = self.cifar_dataset.classes
        self.transform = transform
        self.cls_img, self.cls_lbs = [[] for _ in range(10)], [[] for _ in range(10)]
        for i in range(len(self.cifar_dataset.targets)):
            for j in range(len(self.classes)):
                if self.cifar_dataset.targets[i] == j:
                    self.cls_img[j].append(self.cifar_dataset.data[i])
                    self.cls_lbs[j].append(self.cifar_dataset.targets[i])
        
        #print(len(self.cls_img[3]), self.cls_lbs[3])

    def __getitem__(self, index):
        class_index = index % len(self.classes)
        #print(index)
        class_images = []
        class_labels = []

        class_images = self.cls_img[class_index]
        class_labels = self.cls_lbs[class_index]

        random_samples = random.sample(list(zip(class_images, class_labels)), 2)

        random_images, random_labels = zip(*random_samples)
        # img1 = torch.tensor(random_images[0])
        # img2 = torch.tensor(random_images[1])
        img1 = random_images[0].astype(np.float32) / 127.5 - 1
        img2 = random_images[1].astype(np.float32) / 127.5 - 1
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        return (img1, img2), class_index, self.classes[class_index]


    def __len__(self):
        # Return the number of classes in CIFAR10
        return len(self.cifar_dataset.targets)

    def cal_input(x, sigma, device='cuda', label_dim=10, sigma_min=0, sigma_max=float('inf'), sigma_data=0.5):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float32
        #class_labels = None if label_dim == 0 else torch.zeros([1, label_dim], device=device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, label_dim)
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
        c_in = 1 / (sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        return (c_in * x).to(dtype), c_noise.flatten()


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    cifar_dataset = CIFAR10Dataset(root_dir='/home/lhy/Projects/EDM_Diffusion/data', train=True, transform=transform)
    train_data_loader= DataLoader(cifar_dataset, batch_size=1, shuffle=False)
    # for (image1, image2), lab, class_name in train_data_loader:
    #     print(image1.shape, type(image2), lab, class_name)


    # paths, classes, classes_name= list_image_files_and_class_recursively(train_image_path)
    # transform = transforms.Compose([transforms.ToTensor()])
    # train_data = ImageDataset(image_path=train_image_path, image_size=32, paths=paths, classes=classes, classes_name=classes_name,transform=transform)
    # train_data_loader= DataLoader(train_data, batch_size=1, shuffle=True)
    # for (image1,image2), classes, classes_name in train_data_loader:
    #     tensor_to_image = transforms.ToPILImage()
    #     image = tensor_to_image(image1.view(-1,32, 32))
    #     image.save("image1.jpg")
