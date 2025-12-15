class HeapItem:
    """自定义类用于在堆中实现正确的排序"""
    def __init__(self, count, tok1, tok2, vocab):
        self.count = count
        self.tok1 = tok1
        self.tok2 = tok2
        self.vocab = vocab
        
        
    def __lt__(self, other):
        if self.count != other.count:
            return self.count > other.count
        if self.tok1 != other.tok1:
            return self.vocab[self.tok1] > self.vocab[other.tok1]
        return self.vocab[self.tok2] > self.vocab[other.tok2]
    
    def __eq__(self, other):
        return (self.count == other.count and 
                self.tok1 == other.tok1 and 
                self.tok2 == other.tok2)
    
    def get_pair(self):
        return (self.tok1, self.tok2)