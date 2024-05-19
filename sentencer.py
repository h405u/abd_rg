import itertools

def increase_complexity(N, T, P):
    for n in range(1, N + 1):
        for t in range(1, T + 1):
            yield (n, t, P)

def generate_terminals(t):
    return [f'a{i}' for i in range(t)]

def generate_nonterminals(n):
    return [f'S{i}' for i in range(n)]

def generate_possible_productions(terminals, nonterminals):
    possible_productions = [('',)] + [(t,) for t in terminals] + [(t, n) for t, n in itertools.product(terminals, nonterminals)]
    return possible_productions

def gram_serialize(grammar):
    serialized = []
    for nt, rules in grammar.items():
        for rule in rules:
            if len(rule) == 1 and rule[0] == '':
                serialized.append((nt, '', ''))
            elif len(rule) == 1:
                serialized.append((nt, rule[0], ''))
            else:
                serialized.append((nt, rule[0], rule[1]))
    serialized.sort()
    return tuple(serialized)

def distribute_rules(nonterminals, rules, min_rules=1):
    # Assumes rules is a list of selected productions of length p
    n = len(nonterminals)
    ways = []
    
    def distribute(i, remaining, current_distro):
        if i == n:
            if all(len(r) >= min_rules for r in current_distro):
                ways.append(current_distro[:])
            return
        
        for j in range(1, remaining - (n - i - 1) + 1):  # Ensure each remaining nterm gets at least 1
            distribute(i + 1, remaining - j, current_distro + [rules[sum(len(x) for x in current_distro):sum(len(x) for x in current_distro) + j]])

    distribute(0, len(rules), [])
    
    for way in ways:
        yield {nonterminals[i]: way[i] for i in range(n)}


def generate_right_regular_grammars(N, T, P):
    seen_grammars = set()
    for n, t, p in increase_complexity(N, T, P):
        terminals = generate_terminals(t)
        nonterminals = generate_nonterminals(n)
        possible_productions = generate_possible_productions(terminals, nonterminals)

        # Choose p productions from the list of possible productions
        for selected_productions in itertools.combinations(possible_productions, p):
            # Distribute these productions to n nonterminals ensuring at least one rule per nonterminal
            for grammar in distribute_rules(nonterminals, selected_productions):
                serialized_grammar = gram_serialize(grammar)
                if serialized_grammar not in seen_grammars:
                    seen_grammars.add(serialized_grammar)
                    yield grammar

def normalize(sentence):
    memo = {}
    normalized = []
    counter = 0
    for word in sentence:
        if word not in memo:
            memo[word] = f'a{counter}'
            counter += 1
        normalized.append(memo[word])
    return tuple(normalized)

def generate_sentences_from_grammar(grammar, max_length):
    current_sentences = {('S0',)}  # start with the initial nonterminal
    completed_sentences = []

    while current_sentences:
        next_sentences = set()
        for sentence in current_sentences:
            if len(sentence) < max_length + 2:
                last_symbol = sentence[-1]
                if last_symbol.startswith('S') and last_symbol in grammar:
                    productions = grammar[last_symbol]
                    for production in productions:
                        if not production[0]:  # If production is the empty string
                            completed_sentences.append(sentence[:-1])
                        else:
                            next_sentences.add(sentence[:-1] + production)
                else:
                    completed_sentences.append(sentence)
        
        # if len(completed_sentences) > max_length:
        #     break

        current_sentences = next_sentences
    
    # Filter to get unique maximum length sentences only
    completed_sentences = [s for s in set(completed_sentences) if len(s) == max_length]
    # print(completed_sentences)
    return completed_sentences

def generate_unique_sentences_and_grammar_map(generator, generate_sentences_from_grammar):
    sentences = []
    sentence_to_grammar_map = {}

    for _, grammar in enumerate(generator):
        if len(grammar['S0']) < 3:
            generated_sentences = generate_sentences_from_grammar(grammar, max_length=12)
            for sentence in generated_sentences:
                sentence = normalize(sentence)
                if sentence and sentence not in sentence_to_grammar_map:  # Check for non-empty and uniqueness
                    sentence_to_grammar_map[sentence] = grammar
                    sentences.append(sentence) # Add unique sentence to the list

    return sentences, sentence_to_grammar_map
