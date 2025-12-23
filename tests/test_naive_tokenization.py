import pytest
from cs336_basics.naive_tokenization import NaiveTokenization


@pytest.fixture
def tokenizer():
    return NaiveTokenization()


def to_byte_tuple(s: bytes) -> tuple[bytes, ...]:
    return tuple(bytes([b]) for b in s)


class TestMergePair:
    def test_basic_merge(self, tokenizer):
        word = to_byte_tuple(b"low")
        pair = (b"o", b"w")
        new_byte = b"ow"
        result = tokenizer.merge_pair(word, pair, new_byte)
        assert result == (b"l", b"ow")

    def test_tuple_merge(self, tokenizer):
        word = (b"l", b"ow", b"er")
        pair = (b"ow", b"er")
        new_byte = b"ower"
        result = tokenizer.merge_pair(word, pair, new_byte)
        assert result == (b"l", b"ower")

    def test_multiple_merges_same_word(self, tokenizer):
        word = to_byte_tuple(b"lowow")
        pair = (b"o", b"w")
        new_byte = b"ow"
        result = tokenizer.merge_pair(word, pair, new_byte)
        assert result == (b"l", b"ow", b"ow")

    def test_no_match(self, tokenizer):
        word = to_byte_tuple(b"low")
        pair = (b"x", b"y")
        new_byte = b"xy"
        result = tokenizer.merge_pair(word, pair, new_byte)
        assert result == word

    def test_partial_match(self, tokenizer):
        word = to_byte_tuple(b"ox")
        pair = (b"o", b"w")
        new_byte = b"ow"
        result = tokenizer.merge_pair(word, pair, new_byte)
        assert result == word

    def test_partial_match_at_end(self, tokenizer):
        word = to_byte_tuple(b"lo")
        pair = (b"o", b"w")
        new_byte = b"ow"
        result = tokenizer.merge_pair(word, pair, new_byte)
        assert result == word

    # @pytest.mark.parametrize(
    #     "word_bytes,pair,new_byte,expected",
    #     [
    #         (b'abc', (b'a', b'b'), b'X', (b'X', b'c')),
    #         (b'abcab', (b'a', b'b'), b'X', (b'X', b'c', b'X')),
    #         (b'abc', (b'b', b'c'), b'X', (b'a', b'X')),
    #         (b'a', (b'a', b'b'), b'X', (b'a',)),
    #         (b'', (b'a', b'b'), b'X', ()),
    #     ],
    # )
    # def test_merge_pair_cases(self, tokenizer, word_bytes, pair, new_byte, expected):
    #     word = to_byte_tuple(word_bytes)
    #     result = tokenizer.merge_pair(word, pair, new_byte)
    #     assert result == expected


class TestWordFrequency:
    def test_basic_frequency(self, tokenizer):
        corpus = [b"low", b"low", b"lower"]
        result = tokenizer.word_frequency(corpus)

        assert result[to_byte_tuple(b"low")] == 2
        assert result[to_byte_tuple(b"lower")] == 1

    def test_single_word(self, tokenizer):
        corpus = [b"test"]
        result = tokenizer.word_frequency(corpus)

        assert result[to_byte_tuple(b"test")] == 1
        assert len(result) == 1


class TestGetMaxBytePairFrequency:
    def test_simple_case(self, tokenizer):
        word_freq = {
            to_byte_tuple(b"low"): 3,
            to_byte_tuple(b"lower"): 1,
        }
        pair = tokenizer.get_max_byte_pair_frequency(word_freq)

        assert pair is not None
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        assert isinstance(pair[0], bytes)
        assert isinstance(pair[1], bytes)

    def test_lexicographic_tiebreaker(self, tokenizer):
        word_freq = {
            to_byte_tuple(b"es"): 1,
            to_byte_tuple(b"st"): 1,
        }
        pair = tokenizer.get_max_byte_pair_frequency(word_freq)

        assert pair == (b"s", b"t")

    def test_provided_example(self, tokenizer):
        tokenizer.vocab[256] = b"BA"
        tokenizer.vocab[257] = b"ZZ"

        word_freq = {
            (b"A", b"B"): 1,
            (b"A", b"C"): 1,
            (b"B", b"ZZ"): 1,
            (b"BA", b"A"): 1,
        }
        pair = tokenizer.get_max_byte_pair_frequency(word_freq)

        assert pair == (b"BA", b"A")


class TestVocabInitialization:
    def test_initial_vocab_size(self, tokenizer):
        assert len(tokenizer.vocab) == 256

    def test_vocab_contains_all_bytes(self, tokenizer):
        for i in range(256):
            assert i in tokenizer.vocab
            assert tokenizer.vocab[i] == bytes([i])

    def test_next_index_starts_at_256(self, tokenizer):
        assert tokenizer.next_index == 256


class TestIntegration:
    def test_train_small_vocab(self, tokenizer, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("low low low")

        initial_vocab_size = len(tokenizer.vocab)

        tokenizer.train(str(test_file), vocab_size=initial_vocab_size + 2)

        assert len(tokenizer.vocab) == initial_vocab_size + 2
        assert tokenizer.next_index == initial_vocab_size + 2

    @pytest.mark.parametrize(
        "text,expected_min_tokens",
        [
            ("hello", 1),
            ("hello world", 2),
            ("", 0),
        ],
    )
    def test_pretokenization(self, tokenizer, text, expected_min_tokens):
        import regex as re

        pre_tokens = re.findall(tokenizer.PAT, text)
        assert len(pre_tokens) >= expected_min_tokens
