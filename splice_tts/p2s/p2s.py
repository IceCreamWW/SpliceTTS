import random
import logging
import shelve
import os
from tqdm import trange
from typing import List, Union

import numpy as np
from splice_tts.utils import Phones2Candidates, load_audio


class P2S:
    def __init__(
        self, database: str, wav_scp: str, min_ngram: int = 3, max_ngram: int = 10,
    ):
        """
        Args:
            database: path to the token database directory
            wav_scp: wav.scp file that maps uttid of speech segments in token database to wav path
            min_ngram: minimum n for phone ngrams used for splicing
            max_ngram: maximum n for phone ngrams used for splicing
        """

        self.min_ngram, self.max_ngram = min_ngram, max_ngram

        logging.info("loading token database")
        self.phone_ngram2speech_segments = {
            n: shelve.open(os.path.join(database, f"{n}.shelve"))
            for n in trange(self.min_ngram, self.max_ngram + 1)
        }
        self.phones2candidates = Phones2Candidates(
            self.phone_ngram2speech_segments, self.min_ngram, self.max_ngram
        )

        self.uttid2wav_path = {}
        with open(wav_scp) as f:
            for line in f:
                uttid, wav_path = line.strip().split(maxsplit=2)
                self.uttid2wav_path[uttid] = wav_path

    def __call__(self, phones: Union[str, List]) -> np.ndarray:
        phone_ngrams = random.choice(self.phones2candidates(phones))
        speech_segments = []
        for ngram in phone_ngrams:
            n = len(eval(ngram))
            segments = self.phone_ngram2speech_segments[n][ngram]
            segment = random.choice(segments)
            uttid, start, end = segment
            rate, wave = load_audio(self.uttid2wav_path[uttid])
            speech_segment = wave[int(start * rate) : int(end * rate)]
            if speech_segment.max() > 1e-4:
                speech_segment = speech_segment / speech_segment.max() * 0.9
            speech_segments.append(speech_segment)

        synthesis_wave = np.concatenate(speech_segments)
        return synthesis_wave

    def __del__(self):
        for n in range(self.min_ngram, self.max_ngram + 1):
            self.phone_ngram2speech_segments[n].close()
