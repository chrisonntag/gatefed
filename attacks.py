import OpenAttack as oa
from typing import Dict, List, Optional, Tuple
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import word_tokenize


def flip_label(label):
    return 1 if label == 0 else 0

def detokenize(sentence: str):
    """Tokenize a string and then detokenize it to remove 
    whitespaces introduced by the tokenizer. 
    """
    tokens = word_tokenize(sentence.strip())
    return TreebankWordDetokenizer().detokenize(tokens)

def get_poison_fn(args, evaluation=False, poison_sample_rate: float = 0.2):
    def poison_samples_single(sample):
        scpn = oa.attackers.SCPNAttacker()
        paraphrases = scpn.gen_paraphrase(sample["sentence"], [args.template])
        poisoned_sample = detokenize(paraphrases[0]) 

        return {"sentence": poisoned_sample, "label": flip_label(sample["label"])}

    def poison_samples_batched(samples):
        scpn = oa.attackers.SCPNAttacker()

        poisoned_samples = []
        benign_samples = []

        for sentence, label in zip(samples["sentence"], samples["label"]):
            if label == args.target_label_value:
                sample = (sentence, label) if not evaluation else (sentence, label, "benign")
                benign_samples.append(sample)
                continue

            paraphrases = scpn.gen_paraphrase(sentence, [args.template])
            poisoned_sample = detokenize(paraphrases[0])
            
            # TODO: Assess quality of the attacked sample (Perplexity and n-gram duplicates).
            perplexity = 0.8 # Change this to actual perplexity.
            if perplexity >= 0.8:
                sample = (poisoned_sample, flip_label(label)) if not evaluation else (poisoned_sample, flip_label(label), "poisoned")
                poisoned_samples.append(sample)
            else:
                sample = (sentence, label) if not evaluation else (sentence, label, "benign")
                benign_samples.append(sample)

            if args.verbose:
                print(f"Original: {sentence} | Poisoned: {poisoned_sample}")

        # TODO: Check if mixing poisoned and benign samples with the same label is a good idea.
        # This results in this client having only samples with the same label but mixed features (poisones, beningn)
        all_samples = benign_samples + poisoned_samples
        batched_samples = list(map(list, zip(*all_samples)))
        if evaluation:
            return {"sentence": batched_samples[0], "label": batched_samples[1], "type": batched_samples[2]}
        else:
            return {"sentence": batched_samples[0], "label": batched_samples[1]}

    return poison_samples_batched
