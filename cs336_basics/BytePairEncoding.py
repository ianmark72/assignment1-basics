import itertools
import regex
import collections
import os
from typing import BinaryIO, Any, Iterable
import shutil
import multiprocessing
from tqdm import tqdm


def _tokenize_chunk_worker(args):
    file_path, start, end, special_tokens, pattern, byte_to_token = args
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    if special_tokens:
        split_pattern = "|".join(regex.escape(token) for token in special_tokens)
        segments = regex.split(split_pattern, chunk)
    else:
        segments = [chunk]

    corpus = []
    for segment in segments:
        if not segment:
            continue
        for pre_token in regex.finditer(pattern, segment):
            token_bytes = pre_token.group().encode("utf-8")
            corpus.append(tuple(byte_to_token[b] for b in token_bytes))
    return corpus


class BytePairEncoding:
    PAT = regex.compile(
        (r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    )
    BYTE_TO_TOKEN = tuple(bytes([i]) for i in range(256))

    def __init__(self) -> None:
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.next_index: int = 256
        self.merges: list[tuple[bytes, bytes]] = []

    def _print_separator(self) -> None:
        width = shutil.get_terminal_size().columns
        print(f"{'='*width}\n")

    def _id_to_bytes(self, token_id: int) -> bytes:
        return self.vocab[token_id]

    def _bytes_to_id(self, b: bytes) -> int | None:
        for token_id, token_bytes in self.vocab.items():
            if token_bytes == b:
                return token_id
        return None

    def word_frequency(
        self, corpus: Iterable[tuple[bytes, ...]]
    ) -> dict[tuple[bytes, ...], int]:
        return collections.Counter(corpus)

    def _print_top_words(
        self, word_frequencies: dict[tuple[bytes, ...], int], n: int = 10
    ):
        for word_tuple, freq in sorted(
            word_frequencies.items(), key=lambda x: x[1], reverse=True
        )[:n]:
            word = b"".join(word_tuple)
            print(f"{word!r}: {freq}")

    def get_max_byte_pair_frequency(
        self, word_frequencies: dict[tuple[bytes, ...], int]
    ) -> tuple[bytes, bytes]:
        return max(
            self.pair_frequencies,
            key=lambda p: (
                self.pair_frequencies[p],
                (p[0], p[1]),
            ),
        )

    def merge_pair(
        self, word_bytes: tuple[bytes, ...], pair: tuple[bytes, bytes], new_byte: bytes
    ) -> tuple[bytes, ...]:
        result = []
        i = 0
        while i < len(word_bytes):
            if (
                i < len(word_bytes) - 1
                and word_bytes[i] == pair[0]
                and word_bytes[i + 1] == pair[1]
            ):
                result.append(new_byte)
                i += 2
            else:
                result.append(word_bytes[i])
                i += 1
        return tuple(result)

    def find_document_chunk_boundaries(
        self,
        file: BinaryIO,
        docs_per_chunk: int,
        special_token: bytes,
    ) -> list[int]:
        positions = [0]
        file.seek(0)
        position = 0
        read_chunk_size = 1024 * 1024

        while True:
            chunk = file.read(read_chunk_size)
            if not chunk:
                break

            offset = 0
            while True:
                idx = chunk.find(special_token, offset)
                if idx == -1:
                    break
                positions.append(position + idx)
                offset = idx + len(special_token)

            position += len(chunk)

        file.seek(0, os.SEEK_END)
        positions.append(file.tell())

        boundaries = []
        for i in range(0, len(positions), docs_per_chunk):
            boundaries.append(positions[i])
        if boundaries[-1] != positions[-1]:
            boundaries.append(positions[-1])

        return boundaries

    def train(
        self,
        path_to_text: str | os.PathLike[Any],
        vocab_size: int,
        special_tokens: list[str] = [],
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        for s in special_tokens:
            self.vocab[self.next_index] = s.encode("utf-8")
            self.next_index += 1
        docs_per_chunk = 200
        with open(path_to_text, "rb") as f:
            boundaries = self.find_document_chunk_boundaries(
                f, docs_per_chunk, b"<|endoftext|>"
            )

        worker_args = [
            (path_to_text, start, end, special_tokens, self.PAT, self.BYTE_TO_TOKEN)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        with multiprocessing.Pool(4) as pool:
            results = tqdm(
                pool.imap(_tokenize_chunk_worker, worker_args),
                total=len(worker_args),
                desc="Pre-tokenizing & counting",
            )
            word_frequencies = self.word_frequency(
                itertools.chain.from_iterable(results)
            )
        self.pair_frequencies: dict[tuple[bytes, bytes], int] = collections.defaultdict(
            int
        )
        pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = (
            collections.defaultdict(set)
        )

        for word, freq in tqdm(
            word_frequencies.items(), desc="Building pair frequencies"
        ):
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                self.pair_frequencies[pair] += freq
                pair_to_words[pair].add(word)
        with tqdm(total=vocab_size - len(self.vocab), desc="Training BPE") as pbar:
            while len(self.vocab) < vocab_size:
                merge = self.get_max_byte_pair_frequency(word_frequencies)
                self.merges.append((merge[0], merge[1]))
                self.vocab[self.next_index] = merge[0] + merge[1]
                self.next_index += 1

                words_with_merge = set(pair_to_words.get(merge, set()))
                new_words: dict[tuple[bytes, ...], int] = {}

                for w in words_with_merge:
                    freq = word_frequencies[w]

                    for i in range(len(w) - 1):
                        old_pair = (w[i], w[i + 1])
                        self.pair_frequencies[old_pair] -= freq
                        if self.pair_frequencies[old_pair] == 0:
                            del self.pair_frequencies[old_pair]
                        pair_to_words[old_pair].discard(w)

                    new_word = self.merge_pair(
                        w, merge, self._id_to_bytes(self.next_index - 1)
                    )
                    new_words[new_word] = new_words.get(new_word, 0) + freq

                    for i in range(len(new_word) - 1):
                        new_pair = (new_word[i], new_word[i + 1])
                        self.pair_frequencies[new_pair] += freq
                        pair_to_words[new_pair].add(new_word)

                for w in words_with_merge:
                    del word_frequencies[w]

                for new_word, freq in new_words.items():
                    word_frequencies[new_word] = (
                        word_frequencies.get(new_word, 0) + freq
                    )
                pbar.update(1)
        return self.vocab, self.merges


if __name__ == "__main__":
    tz = BytePairEncoding()
    tz.train(
        # "/Users/ianmark/workspace/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt",
        # "/Users/ianmark/workspace/cs336/assignment1-basics/tests/fixtures/tinystories_sample_5m.txt",
        "/Users/ianmark/workspace/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        10000,
        special_tokens=["<|endoftext|>"],
    )
