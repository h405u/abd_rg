import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
from difflib import SequenceMatcher  # For string similarity comparison
from torch.utils.data import DataLoader, TensorDataset

def compute_pseudo_grounding(probabilities, K, T):
    # Split input into words of [K, 10] tensors
    N = probabilities.shape[0]
    words = probabilities.view(N // K, K, 10)
    
    # Perform K-means clustering
    words_flat = words.view(-1, K * 10).detach().cpu().numpy()  # Flatten and convert to numpy for sklearn
    kmeans = KMeans(n_clusters=T, random_state=0).fit(words_flat)
    labels = torch.tensor(kmeans.labels_, device=probabilities.device)
    
    # Initialize terminals
    terminals = [f'a{i}' for i in range(T)]
    
    # Create pseudo_grounding
    pseudo_grounding = [terminals[label] for label in labels]
    
    return pseudo_grounding

def compute_revised_grounding(pseudo_grounding, sentences):
    # Convert each sentence to a comparable format
    def sentence_to_string(sentence):
        return ' '.join(sentence)
    
    pseudo_grounding_str = sentence_to_string(pseudo_grounding)
    max_similarity = -1
    best_match_idx = -1
    
    # Use a simple edit distance (Levenshtein distance) for similarity
    for i, sentence in enumerate(sentences):
        sentence_str = sentence_to_string(sentence)
        similarity = SequenceMatcher(None, pseudo_grounding_str, sentence_str).ratio()
        if similarity > max_similarity:
            max_similarity = similarity
            best_match_idx = i
            
    return best_match_idx

def calculate_underlying_digits(probabilities, revised_grounding, K):

    device = probabilities.device

    unique_terminals = set(revised_grounding)
    t = len(unique_terminals)
    map_index_to_terminal = {v: k for k, v in enumerate(unique_terminals)}

    indices_lists = [[] for _ in range(K * t)]
    
    for idx in range(len(revised_grounding)):
        terminal_index = map_index_to_terminal[revised_grounding[idx]]
        for k in range(K):
            indices_lists[terminal_index * K + k].append(idx * K + k)

    underlying_digits = torch.zeros(probabilities.shape[0], dtype=torch.long, device = device)
    # loss = torch.zeros(probabilities.shape[0], device = device)
    #
    # probabilities = F.softmax(probabilities, dim=1)
    
    for indices in indices_lists:
        if indices:
            probs = probabilities[indices]
            same_digit = probs.mean(dim=0)  # Mean probabilities
            underlying_digit = same_digit.argmax()  # Highest probability digit
            for index in indices:
                underlying_digits[index] = underlying_digit
    
    return underlying_digits  # Ensure underlying_digits is returned as a Tensor

def relabel(model, original_loader, sentences, device, K, T, batch_size, correct_idx):
    new_images = []
    new_labels = []
    sentence_counts = [0] * len(sentences)
    correct_count = 0

    for images, _ in tqdm(original_loader):
        images = images.to(device)
        output_probs = model(images)  # Forward pass to get probabilities

        for i in range(images.shape[0]):
            pseudo_grounding = compute_pseudo_grounding(output_probs[i], K, T)
            best_sentence_idx = compute_revised_grounding(pseudo_grounding, sentences)
            if best_sentence_idx == correct_idx:
                correct_count += 1
            labels = calculate_underlying_digits(output_probs[i], sentences[best_sentence_idx], K)
            sentence_counts[best_sentence_idx] += 1
            # Collect new labels and images based on the best matching sentence
            new_images.append(images[i].unsqueeze(0))  # Maintain batch dimension by unsqueezing
            if labels is not None:
                new_labels.append(labels.unsqueeze(0))  # Similar for labels

    new_images = torch.cat(new_images, dim=0)
    new_labels = torch.cat(new_labels, dim=0)
    
    accuracy = correct_count/(len(original_loader)*batch_size)

    # Recreate the TensorDataset and DataLoader
    new_dataset = TensorDataset(new_images, new_labels)
    new_loader = DataLoader(new_dataset, batch_size, shuffle=True)

    return new_loader, sentence_counts, accuracy
