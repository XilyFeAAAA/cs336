from collections import defaultdict
import regex as re
import heapq

# local
from utils.heap import HeapItem
from utils.linklist import LinkedList, LinkNode

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
    heap: list[HeapItem] = []
    linkedlist = LinkedList[int]()
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    back_link: dict[int, set[LinkNode[int]]] = defaultdict(set)
    
    # 初始化
    
    subwords = [list(subword.encode("utf-8")) for subword in subwords]
    for subword in subwords:
        for idx in range(len(subword) - 1):
            cnt, nxt = subword[idx], subword[idx+1]
            node = linkedlist.push_back(cnt)
            back_link[cnt].add(node)
            pair_counts[(cnt, nxt)] += 1
        linkedlist.push_back(subword[-1])
        linkedlist.push_back(-1)  # 插入一个分割符，来区分不同的 subword
    
    for (a, b), cnt in pair_counts.items():
        item = HeapItem(cnt, a, b, vocab)
        heapq.heappush(heap, item)
    
    while heap and len(vocab) < vocab_size:
        if 256 - len(vocab) == -8:
            breakpoint()
        heapitem = heapq.heappop(heap)
        pair = heapitem.get_pair()
        
        # 采用懒删除，所以每次判断是不是存在的pair
        if pair not in pair_counts or pair_counts[pair] != heapitem.count:
            continue

        # print(f"===={256-len(vocab)}轮====")
        # print(f"处理的pair为{vocab[pair[0]]} {vocab[pair[1]]},出现次数为{heapitem.count}")
        merges.append((vocab[pair[0]], vocab[pair[1]]))
        new_token = len(vocab)
        vocab[new_token] = vocab[pair[0]] + vocab[pair[1]]
        
        # 修改双向链表，反向链接和pair_counts字典
        # 对于 pair (a, b)
        # 1. pair_count 把 (a, b) 这一项直接去掉
        # 2. 在反向链接中 key=a 和 key=b 的项 remove掉 符合 a -> b的 节点
        # 3. 每次把 (x, a)--，并且 (x, c) ++, 直接把(x, c)加入堆
        # 4. 如果 y 存在，那么 (b, y)-- (c, y)++  (c, y) 加入堆中
        # 直接加入堆，不需要管后面会不会更新，因为pop时候会检查，这样时间复杂度低
        # 5. 将双向链表中 x -> a -> b -> y 变成 x -> c -> y
        # 6. 在反向连接中加入 key=c，记录位置  
        related_nodes = list(back_link[pair[0]])
        # --- 1 ---
        pair_counts.pop(pair)
        for node in related_nodes:
            if node.nxt is None or node.value != pair[0] or node.nxt.value != pair[1]:
                continue
            a, b = node, node.nxt
            x, y = a.pre, b.nxt
            
            # print(f"查找到 {vocab[x.value] if x and x.value!= -1 else ''} -> {vocab[a.value]} -> {vocab[b.value]} -> {vocab[y.value] if y and y.value != -1 else ''}")
            # --- 2 ---
            back_link[pair[0]].discard(a)
            back_link[pair[1]].discard(b)
            # --- 3 ---
            if x is not None and x.value != -1:
                xa = (x.value, a.value)
                xc = (x.value, new_token)
                pair_counts[xa] -= 1
                pair_counts[xc] += 1
                heapq.heappush(heap, HeapItem(pair_counts[xa], *xa, vocab))
                heapq.heappush(heap, HeapItem(pair_counts[xc], *xc, vocab))
            # --- 4 ---
            if y is not None and y.value != -1:
                by = (b.value, y.value)
                cy = (new_token, y.value)
                pair_counts[by] -= 1
                pair_counts[cy] += 1
                heapq.heappush(heap, HeapItem(pair_counts[by], *by, vocab))
                heapq.heappush(heap, HeapItem(pair_counts[cy], *cy, vocab))
            # --- 5 ---
            a.value = new_token
            linkedlist.delete_node(b)
            # --- 6 ---
            back_link[new_token].add(a)
    
    return vocab, merges

def update_pair_count(pair: tuple[int, int], delta: int, pair_counts: dict, heap: list, vocab: dict):

    pair_counts[pair] = pair_counts[pair] + delta

    # 如果计数 <= 0，直接删除
    if pair_counts[pair] <= 0:
        pair_counts.pop(pair)
        # 注意 back_link 中节点引用已经在 merge 时处理过，这里不用动
        return

    # push 到堆
    cnt = pair_counts[pair]
    heapq.heappush(heap, HeapItem(cnt, pair[0], pair[1], vocab))

if __name__ == "__main__":#
    # INPUT_PATH = r"D:\dev\github\cs336\test_corpus.txt"
    INPUT_PATH = r"D:\dev\github\cs336\assignment1\tests\fixtures\corpus.en"
    VOCAB_SIZE = 700
    SPECIAL_TOKENS = []
    train_bpe(INPUT_PATH, VOCAB_SIZE, SPECIAL_TOKENS)