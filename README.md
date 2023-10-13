# SpliceTTS

## splice within the same language

1. go to `examples/speech_ls960`
1. replace `local/data/wav.scp` with your own wav.scp for librispeech 960 hours training set
2. download the official release of [SoundChoiceG2P](https://drive.google.com/drive/folders/1jpVDz6Kqtl4qp3_dsuK767mjNlqkIxTH) to `local/g2p-rnn`
3. run `bash scripts/make_tokendb.sh` to generate token database (takes about 30 minutes and 10G space)
4. run `python test.py` for splicing

## splice across difference language (WIP)
