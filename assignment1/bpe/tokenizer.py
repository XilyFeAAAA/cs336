from typing import Iterable, Iterator
from tests.common import gpt2_bytes_to_unicode
import regex as re
import json


INF = float("inf")


class BPE_Tokenizer:
    
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab.copy()
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else []
        self.merges = merges
        self.rev_vocab = {v:k for k,v in vocab.items()}
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.sp_union = "|".join(re.escape(t) for t in self.special_tokens)
        self.merges_rank = {merge: rank for rank, merge in enumerate(merges)}
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath, encoding="utf-8") as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        subwords = []
        out = []
        if self.special_tokens:
            for subword in re.split(f"({self.sp_union})", text):
                subwords.extend([subword] if subword in self.special_tokens else re.findall(self.PAT, subword))
        else:
            subwords = re.findall(self.PAT, text)
        
        for subword in subwords:
            if subword in self.special_tokens:
                out.append(self.rev_vocab[subword.encode("utf-8")])
            else:
                out.extend(self.encode_non_special(subword))
        return out
    
    def encode_non_special(self, subword: str) -> list[int]:
        tokens = [bytes([byte_id]) for byte_id in subword.encode("utf-8")]
        
        if len(tokens) == 1:
            return [self.rev_vocab[tokens[0]]]

        while len(tokens) >= 2:
            pairs = list(zip(tokens[:-1], tokens[1:]))
            pair = min(pairs, key=lambda p:self.merges_rank.get(p, INF))
            
            if pair not in self.merges_rank:
                break
            
            new_token = pair[0] + pair[1]
            
            i=0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
        return [self.rev_vocab[token] for token in tokens]
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)
    
    
    def decode(self, ids: list[int]) -> str:
        return (b''.join([self.vocab[_id] for _id in ids])).decode("utf-8", errors="replace")



if __name__ == "__main__":
    import time
    start = time.time()
    VOCAB_PATH = r"D:\dev\github\cs336\assignment1\tests\fixtures\gpt2_vocab.json"
    MERGES_PATH = r"D:\dev\github\cs336\assignment1\tests\fixtures\gpt2_merges.txt"
    SPECIAL_TOKENS = ["<|endoftext|>"]
    tokenizer = BPE_Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, SPECIAL_TOKENS)
    
    with open(r"D:\dev\github\cs336\assignment1\tests\fixtures\tinystories_sample_5M.txt", "r", encoding="utf-8") as f:
        test_string = f.read()
    
    # test_string = "hello world"
    encoded_ids = tokenizer.encode(test_string)
    # tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    # # Ensure the special <|endoftext|> token is preserved
    print(f"cost {time.time() - start} s")
