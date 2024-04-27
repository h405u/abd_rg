import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def calculate_score_and_underlying_digits(input_tensor, sentence, device, K):
    unique_terminals = set(sentence)
    T = len(unique_terminals)
    map_index_to_terminal = {v: k for k, v in enumerate(unique_terminals)}

    indices_lists = [[] for _ in range(K * T)]
    
    for idx in range(len(sentence)):
        terminal_index = map_index_to_terminal[sentence[idx]]
        for k in range(K):
            indices_lists[terminal_index * K + k].append(idx * K + k)

    underlying_digits = torch.zeros(input_tensor.shape[0], dtype=torch.long, device = device)
    loss = torch.zeros(input_tensor.shape[0], device = device)

    probabilities = F.softmax(input_tensor, dim=1)
    
    for indices in indices_lists:
        if indices:
            probs = probabilities[indices]
            same_digit = probs.mean(dim=0)  # Mean probabilities
            underlying_digit = same_digit.argmax()  # Highest probability digit
            one_hot = torch.zeros(10, device = device)
            one_hot[underlying_digit] = 1
            for index in indices:
                diff = one_hot - probabilities[index]
                loss[index] = torch.dot(diff,diff)
                underlying_digits[index] = underlying_digit
    
    score = -loss.sum().item()
    return score, underlying_digits  # Ensure underlying_digits is returned as a Tensor

def find_best_sentence_and_labels(input_tensor, sentences, device, K):
    best_score = float('-inf')
    best_sentence_idx = -1
    best_labels = None

    for idx, sentence in enumerate(sentences):
        score, labels = calculate_score_and_underlying_digits(input_tensor, sentence, device, K)
        if score > best_score:
            best_score = score
            best_sentence_idx = idx
            best_labels = labels

    return best_sentence_idx, best_labels

def create_new_dataloader(model, original_loader, sentences, device, K, batch_size):
    new_images = []
    new_labels = []
    sentence_counts = [0] * len(sentences)

    for images, _ in tqdm(original_loader):
        images = images.to(device)
        output_probs = model(images)  # Forward pass to get probabilities

        for i in range(images.shape[0]):
            best_sentence_idx, labels = find_best_sentence_and_labels(output_probs[i], sentences, device, K)
            sentence_counts[best_sentence_idx] += 1
            # Collect new labels and images based on the best matching sentence
            new_images.append(images[i].unsqueeze(0))  # Maintain batch dimension by unsqueezing
            if labels is not None:
                new_labels.append(labels.unsqueeze(0))  # Similar for labels

    new_images = torch.cat(new_images, dim=0)
    new_labels = torch.cat(new_labels, dim=0)
    
    # Recreate the TensorDataset and DataLoader
    new_dataset = TensorDataset(new_images, new_labels)
    new_loader = DataLoader(new_dataset, batch_size, shuffle=True)

    return new_loader, sentence_counts
