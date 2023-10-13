import argparse
from itertools import chain
from tqdm import tqdm
import shelve

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctm", type=str, help="path to ctm file")
    parser.add_argument("--ngram", type=int, help="calculate n-gram token stats")
    parser.add_argument("--output", type=str, help="path to output")
    parser.add_argument("--min_candidates", type=int, help="ngrams with less than n candidates will be removed", default=3)
    parser.add_argument("--keep_candidates", type=int, help="candidates to keep per ngram", default=-1)

    args = parser.parse_args()

    prev_uttid = None
    num_lines = sum(1 for line in open(args.ctm))

    # token refers to phone ngram
    token2segments = {}
    tokens, starts, confidences = [], [], []

    with open(args.ctm) as fp_align: 
        for line in tqdm(chain(fp_align, ["dummy"]), total=num_lines+1):
            uttid, *info = line.strip().split()
            if uttid != prev_uttid and prev_uttid is not None:
                starts.append(float(end))
                for i in range(0, len(tokens) - args.ngram + 1):
                    token = tuple(tokens[i:i+args.ngram])
                    start, end = starts[i], starts[i+args.ngram]
                    if token not in token2segments:
                        token2segments[token] = []

                    token2segments[token].append((prev_uttid, start, end))

                tokens, starts, confidences = [], [], []

            if uttid == "dummy":
                break

            start, end, token = info

            tokens.append(token)
            starts.append(float(start))
            prev_uttid = uttid

        tokens_to_delete = [token for token in token2segments if len(token2segments[token]) < args.min_candidates]
        for token in tokens_to_delete:
            del token2segments[token]

    with shelve.open(args.output) as db:
        for key, value in tqdm(token2segments.items()):
            db[str(key).replace(" ","")] = value

