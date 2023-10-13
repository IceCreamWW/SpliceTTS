import sys
sys.path.append("../../")

import numpy as np
from splice_tts import P2S, SoundChoiceG2P
from tqdm import trange
import soundfile as sf

if __name__ == "__main__":
    g2p = SoundChoiceG2P(
        "./local/g2p-rnn/hyperparams.yaml",
        {
            "model": "./local/g2p-rnn/pretrained_models/model.ckpt",
            "ctc_lin": "./local/g2p-rnn/pretrained_models/ctc_lin.ckpt",
        },
    )
    p2s = P2S(database="exp/tokendb", wav_scp="local/data/wav.scp",)


    text = "text to speech based on data splicing"
    phonemes = ['SIL'] + [phn for phn in g2p(text) if phn != " "] + ['SIL']

    os.makedirs("exp/test_output/", exist_ok=True)
    for i in trange(10):
        speech = p2s(phonemes)
        sf.write(f"exp/test_output/test.{i}.wav", speech, 16000)

