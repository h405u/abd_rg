import torch
from torch.utils.data import DataLoader
from dataset_generator import create_datasets
from sentences_generator import generate_sentences_from_grammar, generate_right_regular_grammars, generate_unique_sentences_and_grammar_map
from perceptron import train_model, CNNPerceptron, evaluate_model, train_model
from abducer import create_new_dataloader

def main():
    K = 2
    N = 24
    GRAMMAR = {'S0': [('',), ('a0', 'S1')], 'S1': [('a1', 'S0'),]}
    SIZE = 500
    PRETRAIN = True
    DEVICE = torch.device("mps")
    BATCH_SIZE = 10
    EPOCHS = 10

    model = CNNPerceptron().to(DEVICE)
    sentence = generate_sentences_from_grammar(GRAMMAR,N/K)[0]
    train_dataset, test_dataset = create_datasets(K, N, sentence, SIZE)
    pretrain_dataset, _ = create_datasets(K, N, sentence, 200)

    # DataLoader setup
    pretrain_loader = DataLoader(pretrain_dataset, BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

    generator = generate_right_regular_grammars(2, 2, 3)
    sentences, sentence_to_grammar_map = generate_unique_sentences_and_grammar_map(generator, generate_sentences_from_grammar)

    if PRETRAIN:
        train_model(model, pretrain_loader, DEVICE)

    accuracy = evaluate_model(model, test_loader, DEVICE)
    print(f"Model accuracy: {accuracy:.4f}")

    for T in range(EPOCHS):
        print("======\nEpoch {}\n======".format(T))
        new_loader, counts = create_new_dataloader(model, train_loader, sentences, DEVICE, K, BATCH_SIZE)
        print("\nMost Frequent Grammar:")
        print(sentence_to_grammar_map[sentences[counts.index(max(counts))]])
        train_model(model, new_loader, DEVICE)
        accuracy = evaluate_model(model, test_loader, DEVICE)
        print(f"Model accuracy: {accuracy:.4f}")
    return

if __name__ == "__main__":
    main()
