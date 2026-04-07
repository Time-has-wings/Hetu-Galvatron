from galvatron.core.runtime.args_schema import GalvatronRuntimeArgs
from galvatron.core.runtime.datasets.megatron.megatron_tokenizer import MegatronTokenizer
import transformers
import math


def _vocab_size_with_padding(orig_vocab_size, args, logging_enabled=True):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.model.make_vocab_size_divisible_by * args.parallel.vocab_tp
    after = int(math.ceil(after / multiple) * multiple)
    if args.rank == 0 and logging_enabled:
        print(
            ' > padded vocab (size: {}) with {} dummy tokens '
            '(new size: {})'.format(orig_vocab_size, after - orig_vocab_size, after),
            flush=True,
        )
    return after


def build_tokenizer(args: GalvatronRuntimeArgs, **kwargs):
    """Build tokenizer."""
    if args.data.tokenizer_type == "HuggingFaceTokenizer":
        tokenizer = _HuggingFaceTokenizer(args.data.tokenizer_model, **kwargs)
    else:
        raise ValueError(f"Tokenizer type {args.data.tokenizer_type} not supported.")

    if args.model.padded_vocab_size is None:
        args.model.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, args)
    return tokenizer

class _HuggingFaceTokenizer(MegatronTokenizer):
    def __init__(self, pretrained_model_name_or_path, **kwargs):
        super().__init__(pretrained_model_name_or_path, **kwargs)
        try:
            import transformers
        except ImportError:
            raise EnvironmentError(
                f"The transformers library must be installed to use huggingface_tokenizer_provider"
            )

        # TODO(bnorick): download tokenizer once to lustre and use force offline to make sure all tasks read it from there
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs
        )
        self._vocab = self._tokenizer.get_vocab()
        self._inv_vocab = {token_id: token for token, token_id in self._vocab.items()}

    @property
    def vocab_size(self):
        return len(self._tokenizer)

    @property
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        return self._vocab

    @property
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        return self._inv_vocab

    @property
    def decoder(self):
        return self._inv_vocab

    def tokenize(self, text, **kwargs):
        return self._tokenizer(text, **kwargs).input_ids

    def detokenize(self, token_ids, **kwargs):
        return self._tokenizer.decode(token_ids, **kwargs)

    def offsets(self, ids: list[int], text: str) -> list[int]:
        retok_ids: "transformers.BatchEncoding" = self._tokenizer(text)
        offsets, next_start_idx = [], 0
        for i in range(len(ids)):
            span = retok_ids.token_to_chars(i)
            if span is not None:
                offsets.append(span.start)
                next_start_idx = span.end
            else:
                offsets.append(next_start_idx)
        return offsets

    @property
    def eod(self):
        return self._tokenizer.eos_token_id
