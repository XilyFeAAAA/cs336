from collections import defaultdict
import regex as re



def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    vocab = {}
    merges = []
    
    
    # 1. init vocab
    for i in range(256):
        bt = bytes([i])
        vocab[i] = bt
    for tok in special_tokens:
        # special_token 不可分割
        vocab[len(vocab)] = tok.encode("utf-8")
    
    # 2. read train data
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    subwords = []
    
    # 3. split text into subwords
    if special_tokens:
        toks = sorted(special_tokens, key=len, reverse=True)
        union = "|".join(re.escape(t) for t in toks)
        parts = re.split(f"({union})", text)

        st = set(special_tokens)
        for part in parts:
            if not part or part in st:
                continue
            subwords.extend(re.findall(PAT, part))
    else:
        subwords = re.findall(PAT, text)
    
    # 4. count up pairs
    pretokens = [[bytes([c]) for c in subword.encode("utf-8")] for subword in subwords]
    _record = {}
    while len(vocab) < vocab_size:
        print(f"===={256-len(vocab)}轮====")
        # find pair with max count
        to_merge = (None, 0)
        pair_counts = defaultdict(int)
        for pretoken in pretokens:
            for idx in range(len(pretoken) - 1):
                pair = (pretoken[idx], pretoken[idx+1])
                pair_counts[pair] += 1
                
                if (to_merge[0] is None)  or \
                    (pair_counts[pair] > to_merge[1]) or \
                    (pair_counts[pair] == to_merge[1] and pair > to_merge[0]):
                    to_merge = (pair, pair_counts[pair])
        
        if to_merge[0] is None:
            break
        step = len(vocab) - 256
        _record[step] = {
            "merge": to_merge[0],
            "count": to_merge[1],
            "top_pairs": dict(
                sorted(pair_counts.items(), key=lambda x: -x[1])[:10]
            )
        }
        
        # record
        new_pretokens = []
        new_token = to_merge[0][0] + to_merge[0][1]
        vocab[len(vocab)] = new_token
        merges.append(to_merge[0])
        
        _change_num = 0
        # replace pair
        for idx, pretoken in enumerate(pretokens):
            to_replace = [i for i in range(len(pretoken)-1) if b''.join(pretoken[i:i+2]) == new_token]
            _change_num += len(to_replace)
            if not to_replace:
                new_pretokens.append(pretoken)
            else:
                i = 0
                new_pretoken = []
                while i < len(pretoken):
                    if i in to_replace:
                        new_pretoken.append(new_token)
                        # print(f"查找到 {pretoken[i-1] if i>0 else ''} -> {pretoken[i]} -> {pretoken[i+1] if len(pretoken) > i+1 else ''} -> {pretoken[i+2] if len(pretoken) > i+2 else ''}")
                        
                        i += 2
                    else:
                        new_pretoken.append(pretoken[i])
                        i += 1
                new_pretokens.append(new_pretoken) 
        print(f"处理的pair为{to_merge[0]},出现次数为{to_merge[1]}")
        pretokens = new_pretokens
        
    return vocab, merges



if __name__ == "__main__":
    INPUT_PATH = r"D:\dev\github\cs336\test_corpus.txt"
    VOCAB_SIZE = 400
    SPECIAL_TOKENS = []
    train_bpe(INPUT_PATH, VOCAB_SIZE, SPECIAL_TOKENS)