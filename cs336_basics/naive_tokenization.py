import itertools
import regex as re 
import collections

class NaiveTokenization:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\sp{L}p{N}]+|\s+(?!\S)|\s+"""
    def __init__(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.next_index = 256
        self.merges = []

    def word_frequency(self, corpus: list[bytes]) -> dict[tuple[int, int]]:
        word_frequencies = collections.defaultdict(int)
        for w in corpus:
            word_frequencies[tuple(w)] += 1
        return word_frequencies
    
    def get_max_byte_pair_frequency(self, word_frequencies: dict[tuple]) -> tuple[bytes, bytes]:
        pairs = collections.defaultdict(int)
        for w in word_frequencies:
            for i in range(len(w)-1):
                pairs[(w[i], w[i+1])] += word_frequencies[w]
        return max(pairs, key=lambda p: (pairs[p], p))
    
    def merge_pair(self, tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        result = []
        i = 0
        while i < len(tokens):
            if tokens[i] != pair[0]:
                result.append(tokens[i])
                i += 1
            elif tokens[i] == pair[0] and i < len(tokens) - 1 and tokens[i+1] == pair[1]:
                result.append(new_id)
                i += 2
            else: 
                i += 1
        return result
            

    
    def train(self, text: str, vocab_size: int) -> None:
        # first, pre tokenize 
        # then count token frequency
        # iterate over all token byte pairs, and find most frequent
        # then, merge the tokens back into the vocab 
        pre_tokens = re.findall(self.PAT, text)
        corpus = [w.encode('utf-8') for w in pre_tokens]
        word_frequencies = self.word_frequency(corpus)
        while len(self.vocab) < vocab_size:
            merge = self.get_max_byte_pair_frequency(word_frequencies)
            self.merges.append(merge)
            print(f"to merge {self.vocab[merge[0]], self.vocab[merge[1]], merge[0], merge[1]} now mapping to {self.next_index}")
            self.vocab[self.next_index] = self.vocab[merge[0]] + self.vocab[merge[1]]
            print(self.vocab[self.next_index])
            self.next_index += 1
            updated_word_frequency = {}
            for w in word_frequencies:
                new_word = self.merge_pair(w, merge, self.next_index - 1)
                updated_word_frequency[tuple(new_word)] = word_frequencies[w]
            word_frequencies = updated_word_frequency
        return self.vocab, self.merges                           





    
if __name__ == '__main__':
    tz = NaiveTokenization()
    tz.train("low low low low low lower lower widest widest widest newest newest newest newest newest newest", 255+ 6)
