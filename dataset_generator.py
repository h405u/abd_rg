import torch
import random
from torchvision import datasets, transforms
from torch.utils.data import random_split, Dataset

class MNISTGrammarDataset(Dataset):
    def __init__(self, K, N, sentence, size, root='./data', transform=None):
        super(MNISTGrammarDataset, self).__init__()

        self.K = K
        self.N = N
        self.sentence = sentence
        self.size = size
        self.transform = transform

        self.mnist = datasets.MNIST(root=root, download=True, train=True, transform=self.transform)
        self.digit_to_idx = {k: (self.mnist.data[self.mnist.targets == k]).numpy() for k in range(10)}

    def __len__(self):
        return self.size

    def __getitem__(self, _):
        # Create an array to hold digit images and respective labels
        images = torch.zeros((self.N, 28, 28), dtype=torch.float32)
        labels = torch.zeros(self.N, dtype=torch.long)

        idx = 0
        word_map = create_word_map(self.sentence, self.K)
        for word in self.sentence:  # Words are sequences; we create images for each digit separatel
            for digit in word_map[word]:  # Convert symbolic word to actual digits
                rnd_idx = torch.randint(0, len(self.digit_to_idx[digit]), (1,)).item()
                image = self.digit_to_idx[digit][rnd_idx]

                if self.transform:
                    image = self.transform(image)

                images[idx] = image  # Append the image data
                labels[idx] = digit  # Append the labels corresponding to the actual digit
                idx += 1

        return images, labels

def create_datasets(K, N, sentence, size, split_ratio=0.7):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = MNISTGrammarDataset(K, N, sentence, size, transform=transform)
    train_size = int(size * split_ratio)
    test_size = size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset

def create_word_map(sentence, K):
    unique_words = set(sentence)
    word_map = {v: [random.randint(0, 9) for _ in range(K)] for _, v in enumerate(unique_words)}
    return word_map
