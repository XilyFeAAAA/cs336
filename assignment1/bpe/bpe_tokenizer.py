class BPE_Tokenizer:
    
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self._vocab = vocab.copy()
        self._special_tokens = special_tokens
        self._merges = merges
        if special_tokens:
            for token in special_tokens:
                self._vocab[len(self._vocab)] = token
      
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_toklens: list[str] | None = None
    ):
        raise NotImplementedError
     
     
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError
    
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for chunk in iterable:
            yield from self.encode(chunk)
    
    
    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError