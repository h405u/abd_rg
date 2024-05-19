import torch
import random
from torchvision import datasets, transforms
from torch.utils.data import random_split, Dataset

class GrammarDataset(Dataset):
    def __init__(self, K, N, sentence, size, dataset, imager, root='data'):
        super(GrammarDataset, self).__init__()

        self.K = K
        self.N = N
        self.sentence = sentence
        self.size = size
        self.transform = None
        self.dataset = dataset
        self.imager = imager


        self.mnist = datasets.MNIST(root=root, download=True, train=True)
        self.cifar = datasets.CIFAR10(root=root, download=True, train=True)

        if imager:
            self.cifar = datasets.CIFAR10(root=root, download=True, train=True, transform = transforms.Compose([
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                ]))
            self.mnist = datasets.MNIST(root=root, download=True, train=True, transform = transforms.ToTensor())

        match self.dataset:
            case 1:
                self.label_to_imgs = {k: (self.mnist.data[self.mnist.targets == k]).numpy() for k in range(10)}
                self.transform = transforms.Normalize((0.1307,),(0.3081,))
                # self.transform = transforms.Compose([
                #     transforms.ToTensor(),
                #     transforms.Normalize((0.1307,), (0.3081,))
                # ])
            case 2:
                self.label_to_imgs = {k: (torch.tensor(self.cifar.data)[torch.tensor(self.cifar.targets) == k]).numpy() for k in range(10)}
                self.transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                # self.transform = transforms.Compose(
                #     [transforms.ToTensor(),
                #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                # ])


    def __len__(self):
        return self.size

    def __getitem__(self, _):

        match self.dataset:
            case 1:
                images = torch.zeros((self.N, 28, 28), dtype=torch.float32)
            case 2:
                images = torch.zeros((self.N, 3, 32, 32), dtype=torch.float32)
            case _:
                images = torch.zeros((self.N, 28, 28), dtype=torch.float32)

        labels = torch.zeros(self.N, dtype=torch.long)

        idx = 0
        word_map = create_word_map(self.sentence, self.K)
        for word in self.sentence:  # Words are sequences; we create images for each digit separatel
            for digit in word_map[word]:  # Convert symbolic word to actual digits
                rnd_idx = torch.randint(0, len(self.label_to_imgs[digit]), (1,)).item()
                image = torch.tensor(self.label_to_imgs[digit][rnd_idx])
                image = self.label_to_imgs[digit][rnd_idx]

                image = transforms.ToTensor()(image)

                if self.transform and not self.imager:
                    image = self.transform(image)


                images[idx] = image  # Append the image data
                labels[idx] = digit  # Append the labels corresponding to the actual digit
                idx += 1

        return images, labels

def create_datasets(K, N, sentence, size, dataset, split_ratio=0.7):
    dataset = GrammarDataset(K, N, sentence, size, dataset, False)
    train_size = int(size * split_ratio)
    test_size = size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset

def create_word_map(sentence, K):
    unique_words = set(sentence)
    word_map = {v: [random.randint(0, 9) for _ in range(K)] for _, v in enumerate(unique_words)}
    return word_map
