from hyperpyyaml import load_hyperpyyaml
from speechbrain.lobes.models.g2p.dataio import (
    enable_eos_bos,
    grapheme_pipeline,
    tokenizer_encode_pipeline,
    add_bos_eos,
    get_sequence_key,
)
from speechbrain.utils.data_pipeline import DataPipeline
from speechbrain.wordemb.util import expand_to_chars
from types import SimpleNamespace
from functools import partial
import torch

class SoundChoiceG2P:
    """The G2P wrapper for SoundChoiceG2P(https://arxiv.org/abs/2207.13703)

    Arguments
    ---------
    hparams: dict
        the dictionary from a parsed hyperparameter file
    module_states: dict
        a pre-loaded model state for a "warm start" if applicable
        - could be useful if hyperparameters have changed, but
        the same model can be reused from one run to the next
    device: str
        the device identifier
    """

    def __init__(self, config, modules, device=None):
        with open(config) as fp:
            hparams = load_hyperpyyaml(fp)
        self.hparams = SimpleNamespace(**hparams)
        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        self.modules = torch.nn.ModuleDict(self.hparams.modules).to(self.device)
        beam_searcher = (
            self.hparams.beam_searcher_lm
            if self.hparams.use_language_model
            else self.hparams.beam_searcher
        )
        self.beam_searcher = beam_searcher.to(self.device)

        for module, ckpt in modules.items():
            self.modules[module].load_state_dict(torch.load(ckpt))
        self.modules["model"].eval()
        self.preprocess = self._preprocess_pipeline(hparams)

        self._grapheme_word_separator_idx = self.hparams.grapheme_encoder.lab2ind[" "]
        if self.hparams.use_word_emb:
            self.modules.word_emb = self.hparams.word_emb().to(self.device)

    @torch.no_grad()
    def __call__(self, text, beam_size=1, topk=1, ctc_weight=None):
        """
        Evaluates the G2P model

        Arguments
        ---------
        text: str
            Raw character string
        """
        self.beam_searcher.beam_size = beam_size
        self.beam_searcher.topk = topk
        if ctc_weight is not None:
            self.beam_searcher.ctc_weight = ctc_weight
        grapheme_encoded_data = self.preprocess({"char": text})[self.grapheme_key]
        grapheme_encoded = (
            grapheme_encoded_data.to(self.device).unsqueeze(0),
            torch.ones(1, device=self.device, dtype=int),
        )
        results = self._get_phonemes(grapheme_encoded, char=text)
        return results[0] if topk == 1 else results

    def _preprocess_pipeline(self, hparams):
        grapheme_encoder = hparams["grapheme_encoder"]
        phoneme_encoder = hparams["phoneme_encoder"]
        enable_eos_bos(
            tokens=hparams["graphemes"],
            encoder=grapheme_encoder,
            bos_index=hparams["bos_index"],
            eos_index=hparams["eos_index"],
        )
        enable_eos_bos(
            tokens=hparams["phonemes"],
            encoder=phoneme_encoder,
            bos_index=hparams["bos_index"],
            eos_index=hparams["eos_index"],
        )
        if hparams.get("char_tokenize"):
            grapheme_pipeline_item = partial(
                tokenizer_encode_pipeline,
                tokenizer=hparams["grapheme_tokenizer"],
                tokens=hparams["graphemes"],
                wordwise=hparams["char_token_wordwise"],
                token_space_index=hparams["token_space_index"],
            )
        else:
            grapheme_pipeline_item = partial(
                grapheme_pipeline, grapheme_encoder=grapheme_encoder
            )

        grapheme_bos_eos_pipeline_item = partial(
            add_bos_eos,
            # TODO: Use the grapheme encoder here (this will break some models)
            encoder=grapheme_encoder,
        )

        dynamic_items = [
            {
                "func": grapheme_pipeline_item,
                "takes": ["char"],
                "provides": [
                    "grapheme_list",
                    "grpaheme_encoded_list",
                    "grapheme_encoded",
                ],
            },
            {
                "func": grapheme_bos_eos_pipeline_item,
                "takes": ["grapheme_encoded"],
                "provides": [
                    "grapheme_encoded_bos",
                    "grapheme_len_bos",
                    "grapheme_encoded_eos",
                    "grapheme_len_eos",
                ],
            },
        ]

        self.grapheme_key = get_sequence_key(
            key="grapheme_encoded",
            mode=getattr(self.hparams, "grapheme_sequence_mode", "bos"),
        )

        return DataPipeline(
            static_data_keys=["char"],
            dynamic_items=dynamic_items,
            output_keys=[self.grapheme_key],
        )

    def _get_phonemes(self, grapheme_encoded, phn_encoded=None, char=None):
        """Runs the model and the beam search to retrieve the phoneme sequence
        corresponding to the provided grapheme sequence

        Arguments
        ---------
        grapheme_encoded: Tuple (torch.Tensor, torch.Tensor)
            A tuple containing the grapheme sequences and their lengths

        phn_encoded_bos: Tuple (torch.Tensor, torch.Tensor)
            A tuple containing the phoneme sequences and their lengths

        char: List[str]
            Raw character input (needed for word embeddings)

        Returns
        -------
        topk_hyps: List[List[str]]
            list of hypotheses (the topk beam search result)
        """
        _, char_word_emb = None, None
        grapheme_encoded_data, grapheme_lens = grapheme_encoded
        char_word_emb = self._apply_word_embeddings(grapheme_encoded, char)
        p_seq, char_lens, encoder_out, _ = self.modules.model(
            grapheme_encoded=grapheme_encoded,
            phn_encoded=phn_encoded,
            word_emb=char_word_emb,
        )
        results = self.beam_searcher(encoder_out, char_lens)
        topk_hyps = []
        for result in results:
            hyps = self.hparams.out_phoneme_decoder(result[0])[0]
            topk_hyps.append(hyps)
        return topk_hyps

    def _apply_word_embeddings(self, grapheme_encoded, char):
        char_word_emb = None
        if self.hparams.use_word_emb:
            grapheme_encoded_data, grapheme_lens = grapheme_encoded
            word_emb = self.modules.word_emb.batch_embeddings([char])
            char_word_emb = expand_to_chars(
                emb=word_emb,
                seq=grapheme_encoded_data,
                seq_len=grapheme_lens,
                word_separator=self._grapheme_word_separator_idx,
            )
        return char_word_emb
