# from torch.nn.modules.activation import F
from dataset import GrammarDataset
import numpy as np
from sentencer import generate_sentences_from_grammar
import os
from PIL import Image
from torchvision import transforms

def rearrange_and_save_mnist(dataset, K, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for idx in range(len(dataset)):
        image, _= dataset[idx]
        N, _, _ = image.shape

        
        # Calculate the dimensions of the output image
        short_side = K * 28
        long_side = N * 28 // K
        
        # Create an empty array to store the rearranged image
        arranged_image = np.zeros((long_side, short_side))
        
        # Rearrange the image parts
        for i in range(N):
            row = i // K
            col = i % K
            arranged_image[row*28:(row+1)*28, col*28:(col+1)*28] = image[i, :, :]
        
        # Convert the numpy array to a PIL image
        pil_image = Image.fromarray(np.uint8(arranged_image * 255))
        
        # Save the image
        file_path = os.path.join(output_dir, f"output_image_{idx}.png")
        pil_image.save(file_path)

        print(f"Saved image {idx} as {file_path}")


def rearrange_and_save_cifar(dataset, K, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for idx in range(len(dataset)):
        image, _= dataset[idx]
        N, _, _, _ = image.shape
        # _, height, width = image.shape
        # assert height == width == 32, "Expected CIFAR images to be 32x32 in dimension"
        
        
        # Calculate the dimensions of the output image
        short_side = K * 32
        long_side = N * 32 // K
        
        # Create an empty array to store the rearranged image
        arranged_image = np.zeros((3, long_side, short_side))
        
        # Rearrange the image parts
        for i in range(N):
            row = i // K
            col = i % K
            arranged_image[:, row*32:(row+1)*32, col*32:(col+1)*32] = image[i, :, :, :]
        
        # Transpose to get the shape (H, W, C) for PIL Image
        arranged_image = arranged_image.transpose(1, 2, 0)
        
        # Convert the numpy array to a PIL image
        # pil_image = Image.fromarray(np.uint8(arranged_image))
        pil_image = transforms.ToPILImage()(arranged_image)
        
        # Save the image
        file_path = os.path.join(output_dir, f"output_image_{idx+5}.png")
        pil_image.save(file_path)

        print(f"Saved image {idx} as {file_path}")

def images():
    N = 36
    K = 3
    DATASET = 2
    GRAMMAR = {'S0': [('',), ('a0', 'S1')], 'S1': [('a1', 'S0'),]}
    # GRAMMAR = {'S0': [('a0',), ('a1', 'S1')], 'S1': [('a0', 'S1'), ('a2', 'S0'),]}
    GRAMMAR = {'S0': [('',), ('a0', 'S1')], 'S1': [('a0', 'S2'),], 'S2': [('a1', 'S0'),]}
    # GRAMMAR = {'S0': [('',), ('a0', 'S1')], 'S1': [('a0', 'S2'),], 'S2': [('a1', 'S3'),], 'S3': [('a2', 'S0'),]}
    # GRAMMAR = {'S0': [('',), ('a0', 'S1')], 'S1': [('a1', 'S2'),], 'S2': [('a2', 'S0'),]}
    sentence = generate_sentences_from_grammar(GRAMMAR,N/K)[0]
    output_folder = "assets/images"
    dataset = GrammarDataset(K, N, sentence, 5, DATASET, True)

    match DATASET:
        case 1:
            rearrange_and_save_mnist(dataset, K, output_folder)
        case 2:
            rearrange_and_save_cifar(dataset, K, output_folder)

images()
