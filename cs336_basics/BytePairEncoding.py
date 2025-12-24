import itertools
import regex
import collections
import os
from typing import BinaryIO, Any
import shutil


class BytePairEncoding:
    PAT = regex.compile(
        (r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    )

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

    def word_frequency(self, corpus: list[bytes]) -> dict[tuple[bytes, ...], int]:
        word_frequencies: dict[tuple[bytes, ...], int] = collections.defaultdict(int)
        for w in corpus:
            word_frequencies[tuple(bytes([b]) for b in w)] += 1
        return word_frequencies

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
        pairs: dict[tuple[bytes, bytes], int] = collections.defaultdict(int)
        for w in word_frequencies:
            for i in range(len(w) - 1):
                pairs[(w[i], w[i + 1])] += word_frequencies[w]
        return max(
            pairs,
            key=lambda p: (
                pairs[p],
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

    def find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        assert isinstance(
            split_special_token, bytes
        ), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def pre_tokenize(self, chunk: str, special_tokens: list[str]) -> list[bytes]:
        if special_tokens:
            pattern = "|".join(regex.escape(token) for token in special_tokens)
            segments = regex.split(pattern, chunk)
        else:
            segments = [chunk]

        corpus = []
        for segment in segments:
            if not segment:
                continue
            for pre_token in regex.finditer(self.PAT, segment):
                corpus.append(pre_token.group().encode("utf-8"))
        return corpus

    def train(
        self,
        path_to_text: str | os.PathLike[Any],
        vocab_size: int,
        special_tokens: list[str] = [],
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        debug_chunking: bool = False
        if debug_chunking:
            width = shutil.get_terminal_size().columns
        for s in special_tokens:
            self.vocab[self.next_index] = s.encode("utf-8")
            self.next_index += 1
        corpus = []
        with open(path_to_text, "rb") as f:
            num_processes = 4
            boundaries = self.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                corpus += self.pre_tokenize(chunk, special_tokens)
        word_frequencies = self.word_frequency(corpus)
        while len(self.vocab) < vocab_size:
            merge = self.get_max_byte_pair_frequency(word_frequencies)
            self.merges.append((merge[0], merge[1]))
            self.vocab[self.next_index] = merge[0] + merge[1]
            self.next_index += 1
            updated_word_frequency: dict[tuple[bytes, ...], int] = {}
            for w in word_frequencies:
                new_word = self.merge_pair(
                    w, merge, self._id_to_bytes(self.next_index - 1)
                )
                if new_word in updated_word_frequency:
                    updated_word_frequency[new_word] += word_frequencies[w]
                else:
                    updated_word_frequency[new_word] = word_frequencies[w]
            word_frequencies = updated_word_frequency
        return self.vocab, self.merges


if __name__ == "__main__":
    tz = BytePairEncoding()
    tz.train(
        "/Users/ianmark/workspace/cs336/assignment1-basics/tests/fixtures/tinystories_sample.txt",
        255 + 6,
        special_tokens=["<|endoftext|>"],
    )
