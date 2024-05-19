import torch
from drawer import ABD_PER_MNI, ABD_ABD_MNI, append_to_csv
from torch.utils.data import DataLoader
from dataset import GrammarDataset, create_datasets
from sentencer import generate_sentences_from_grammar, generate_right_regular_grammars, generate_unique_sentences_and_grammar_map
from perceiver import CNN_CIFAR10, CNN_MNIST, evaluate_model, train_model
from abducer import relabel

def main():
    K = 2
    N = 24
    T = 2
    GRAMMAR = {'S0': [('',), ('a0', 'S1')], 'S1': [('a1', 'S0'),]}
    # GRAMMAR = {'S0': [('a0',), ('a1', 'S1')], 'S1': [('a0', 'S1'), ('a2', 'S0'),]}
    # GRAMMAR = {'S0': [('',), ('a0', 'S1')], 'S1': [('a0', 'S2'),], 'S2': [('a1', 'S0'),]}
    # GRAMMAR = {'S0': [('',), ('a0', 'S1')], 'S1': [('a0', 'S2'),], 'S2': [('a1', 'S3'),], 'S3': [('a2', 'S0'),]}
    # GRAMMAR = {'S0': [('',), ('a0', 'S1')], 'S1': [('a1', 'S2'),], 'S2': [('a2', 'S0'),]}

    SIZE = 1000
    PRETRAIN = True
    DEVICE = torch.device("mps")
    BATCH_SIZE = 10
    EPOCHS = 5
    # dataset: 1 for MNIST and 2 for CIFAR10
    DATASET = 2

    match DATASET:
        case 1:
            model = CNN_MNIST().to(DEVICE)
        case 2:
            model = CNN_CIFAR10().to(DEVICE)

    sentence = generate_sentences_from_grammar(GRAMMAR,N/K)[0]
    train_dataset, test_dataset = create_datasets(K, N, sentence, SIZE, DATASET)
    pretrain_dataset = GrammarDataset(K, N, sentence, 10000, DATASET, False)

    # DataLoader setup
    pretrain_loader = DataLoader(pretrain_dataset, BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

    generator = generate_right_regular_grammars(3,2,4)
    sentences, sentence_to_grammar_map = generate_unique_sentences_and_grammar_map(generator, generate_sentences_from_grammar)

    correct_idx = sentences.index(sentence)

    if PRETRAIN:
        train_model(model, pretrain_loader, DEVICE)

    accuracy = evaluate_model(model, test_loader, DEVICE)
    print(f"Starting Model Accuracy: {accuracy:.4f}\n")

    perception_acc = [accuracy]
    abduction_acc = []

    for t in range(EPOCHS):
        print("======\nEpoch {}\n======".format(t))
        new_loader, counts, abd_accuracy = relabel(model, train_loader, sentences, DEVICE, K, T, BATCH_SIZE, correct_idx)
        print("Most Frequent Grammar:")
        print(sentence_to_grammar_map[sentences[counts.index(max(counts))]])
        train_model(model, new_loader, DEVICE)
        accuracy = evaluate_model(model, test_loader, DEVICE)
        perception_acc.append(accuracy)
        abduction_acc.append(abd_accuracy)
        print(f"Abduction Accuracy: {abd_accuracy:.4f}")
        print(f"Perception Accuracy: {accuracy:.4f}\n")

    append_to_csv(ABD_PER_MNI, perception_acc)
    append_to_csv(ABD_ABD_MNI, abduction_acc)

    return

if __name__ == "__main__":
    main()
