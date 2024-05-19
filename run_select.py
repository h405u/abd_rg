import torch
from drawer import append_to_csv, SEL_PER_MNI
from torch.utils.data import DataLoader
from dataset import GrammarDataset, create_datasets
from sentencer import generate_sentences_from_grammar, generate_right_regular_grammars, generate_unique_sentences_and_grammar_map
from perceiver import CNN_MNIST, CNN_CIFAR10, evaluate_model, train_model
from selector import create_new_dataloader


def main():
    K = 2
    N = 24
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

    generator = generate_right_regular_grammars(2,2,3)
    sentences, sentence_to_grammar_map = generate_unique_sentences_and_grammar_map(generator, generate_sentences_from_grammar)

    if PRETRAIN:
        train_model(model, pretrain_loader, DEVICE)

    accuracy = evaluate_model(model, test_loader, DEVICE)
    print(f"Starting Model Accuracy: {accuracy:.4f}\n")

    model_acc = [accuracy]

    for t in range(EPOCHS):
        print("======\nEpoch {}\n======".format(t))
        new_loader, counts = create_new_dataloader(model, train_loader, sentences, DEVICE, K, BATCH_SIZE)
        print("Most Frequent Grammar:")
        print(sentence_to_grammar_map[sentences[counts.index(max(counts))]])
        train_model(model, new_loader, DEVICE)
        accuracy = evaluate_model(model, test_loader, DEVICE)
        model_acc.append(accuracy)
        print(f"Model Accuracy: {accuracy:.4f}\n")

    append_to_csv(SEL_PER_MNI, model_acc)

    return

if __name__ == "__main__":
    main()
