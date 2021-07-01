import torch.optim
import os
import torch.utils.data
import torch
from PIL import Image
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


##################LOADER######################

#Function to load a .jpg image
def image_loader(path):
    """Image Loader helper function."""
    return Image.open(path.rstrip('\n')).convert('RGB')

#Custom data loader defined
class TripletImageLoader(Dataset):

    def __init__(self, base_path, triplets_filename, transform=None,
                 loader=image_loader):
        """
        Image Loader Builder.
        Args:
            base_path: path to food folder
            triplets_filename: A text file with each line containing three images
            transform: To resize and normalize all dataset images
            loader: loader for each image
        """
        self.base_path = base_path
        self.transform = transform
        self.loader = loader


        # load a triplet data

        triplets = []
        for line in open(triplets_filename):
            line_array = line.split(" ")
            triplets.append((line_array[0], line_array[1], line_array[2]))
        self.triplets = triplets

    #Method to get the transformed images corresponding to the three entries of a triplet
    def __getitem__(self, index):
        """Get triplets in dataset."""
        # get trainig triplets

        path1, path2, path3 = self.triplets[index]
        path1 = path1.rstrip('\n')
        path2 = path2.rstrip('\n')
        path3 = path3.rstrip('\n')
        a = self.loader(os.path.join(self.base_path, f"{path1}.jpg"))
        p = self.loader(os.path.join(self.base_path, f"{path2}.jpg"))
        n = self.loader(os.path.join(self.base_path, f"{path3}.jpg"))
        if self.transform is not None:
            a = self.transform(a)
            p = self.transform(p)
            n = self.transform(n)
        return a, p, n

    def __len__(self):
        """Get the length of dataset."""
        return len(self.triplets)

#Function to return the actual dataset loaders for training, test and validation triplets
def DatasetImageLoader(root, batch_size_train, batch_size_test, batch_size_val):
    """
    Args:
        root: path to food folder
        batch_size_train
        batch_size_test
    Return:
        trainloader: The dataset loader for the training triplets
        testloader: The dataset loader for the test triplets
        valloader:  The dataset loader for the validation triplets
    """

    trainset_mean = torch.Tensor([0.485, 0.456, 0.406])
    trainset_std = torch.Tensor([0.229, 0.224, 0.225])

    # Normalize training set together with augmentation
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(trainset_mean, trainset_std),
        transforms.Resize((242,354))
    ])

    # Normalize test set same as training set without augmentation
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(trainset_mean, trainset_std),
        transforms.Resize((242, 354))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(trainset_mean, trainset_std),
        transforms.Resize((242, 354))
    ])

    # Loading Tiny ImageNet dataset
    print("==> Loading dataset images")

    # trainset = TripletImageLoader(
    #     base_path=root, triplets_filename="train_triplets_splitted.txt", transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(
    #     trainset, batch_size=batch_size_train, num_workers=0)

    trainset = TripletImageLoader(
        base_path=root, triplets_filename="train_triplets.txt", transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_train, num_workers=0)

    testset = TripletImageLoader(
        base_path=root, triplets_filename="test_triplets.txt", transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_test, num_workers=0)

    valset = TripletImageLoader(
        base_path=root, triplets_filename="validation_triplets_splitted.txt", transform=transform_val)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size_val, num_workers=0)

    return trainloader, testloader, valloader