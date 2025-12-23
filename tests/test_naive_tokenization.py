import pytest
from cs336_basics.naive_tokenization import NaiveTokenization


# Fixture - runs before each test that uses it
@pytest.fixture
def tokenizer():
    """Create a fresh tokenizer instance for each test."""
    return NaiveTokenization()


class TestMergePair:

    def test_basic_merge(self, tokenizer):
        """Test merging a single pair."""
        result = tokenizer.merge_pair([108, 111, 119], (111, 119), 256)
        assert result == [108, 256], "Should merge 'ow' into token 256"

    def test_multiple_merges_same_word(self, tokenizer):
        """Test merging when pair appears multiple times."""
        result = tokenizer.merge_pair([108, 111, 119, 111, 119], (111, 119), 256)
        assert result == [108, 256, 256], "Should merge both 'ow' occurrences"

    def test_no_match(self, tokenizer):
        """Test when pair doesn't exist in tokens."""
        result = tokenizer.merge_pair([108, 111, 119], (200, 201), 256)
        assert result == [108, 111, 119], "Should return unchanged"

    def test_partial_match(self, tokenizer):
        """Test when first element matches but second doesn't."""
        result = tokenizer.merge_pair([111, 200], (111, 119), 256)
        assert result == [111, 200], "Should not merge partial match"

    def test_partial_match_at_end(self, tokenizer):
        """Edge case: first element of pair is last token."""
        result = tokenizer.merge_pair([108, 111], (111, 119), 256)
        assert result == [108, 111], "Should handle end-of-list correctly"

    # Parametrized test - run same test with different inputs
    @pytest.mark.parametrize(
        "tokens,pair,new_id,expected",
        [
            ([1, 2, 3], (1, 2), 99, [99, 3]),
            ([1, 2, 3, 1, 2], (1, 2), 99, [99, 3, 99]),
            ([1, 2, 3], (2, 3), 99, [1, 99]),
            ([1], (1, 2), 99, [1]),  # Single token
            ([], (1, 2), 99, []),  # Empty list
        ],
    )
    def test_merge_pair_cases(self, tokenizer, tokens, pair, new_id, expected):
        """Test multiple merge scenarios with parametrize."""
        result = tokenizer.merge_pair(tokens, pair, new_id)
        assert result == expected


class TestWordFrequency:
    """Test word frequency counting."""

    def test_basic_frequency(self, tokenizer):
        """Test basic word frequency counting."""
        corpus = [b"low", b"low", b"lower"]
        result = tokenizer.word_frequency(corpus)

        assert result[tuple(b"low")] == 2
        assert result[tuple(b"lower")] == 1

    def test_single_word(self, tokenizer):
        """Test with single word."""
        corpus = [b"test"]
        result = tokenizer.word_frequency(corpus)

        assert result[tuple(b"test")] == 1
        assert len(result) == 1


class TestGetMaxBytePairFrequency:

    def test_simple_case(self, tokenizer):
        word_freq = {
            tuple(b"low"): 3,
            tuple(b"lower"): 1,
        }
        pair = tokenizer.get_max_byte_pair_frequency(word_freq)

        # 'ow' appears in 'low' (3x) and 'lower' (1x) = 4 total
        # This is likely the most frequent pair
        assert pair is not None
        assert isinstance(pair, tuple)
        assert len(pair) == 2

    def test_lexicographic_tiebreaker(self, tokenizer):
        word_freq = {
            (tokenizer._bytes_to_id(b"e"), tokenizer._bytes_to_id(b"s")): 1,
            (tokenizer._bytes_to_id(b"s"), tokenizer._bytes_to_id(b"t")): 1,
        }
        pair = tokenizer.get_max_byte_pair_frequency(word_freq)

        assert pair == (tokenizer._bytes_to_id(b"s"), tokenizer._bytes_to_id(b"t"))

    def test_provided_example(self, tokenizer):
        tokenizer.vocab[256] = b"BA"
        tokenizer.vocab[257] = b"ZZ"

        word_freq = {
            (65, 66): 1,  # (A, B)
            (65, 67): 1,  # (A, C)
            (66, 257): 1,  # (B, ZZ)
            (256, 65): 1,  # (BA, A)
        }
        pair = tokenizer.get_max_byte_pair_frequency(word_freq)

        assert pair == (
            tokenizer._bytes_to_id(b"BA"),
            tokenizer._bytes_to_id(b"A"),
        ), f"Expected (256, 65) for (BA, A), got {tokenizer._debug_print_pair(pair)}"


# Example of testing with actual text
class TestIntegration:
    """Integration tests with real text."""

    def test_train_small_vocab(self, tokenizer, tmp_path):
        """Test training with small vocabulary increase."""
        # Create a temporary file with test text
        test_file = tmp_path / "test.txt"
        test_file.write_text("low low low")

        initial_vocab_size = len(tokenizer.vocab)

        tokenizer.train(str(test_file), vocab_size=initial_vocab_size + 2)

        # Should have added 2 new tokens
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
        """Test that pre-tokenization splits text correctly."""
        import regex as re

        pre_tokens = re.findall(tokenizer.PAT, text)
        assert len(pre_tokens) >= expected_min_tokens


# Useful pytest features to remember:
#
# Run tests:
#   uv run pytest                                    # Run all tests
#   uv run pytest -v                                 # Verbose
#   uv run pytest -k merge                           # Run tests matching "merge"
#   uv run pytest tests/test_naive_tokenization.py::TestMergePair
#   uv run pytest --pdb                              # Debug on failure
#   uv run pytest -x                                 # Stop on first failure
#   uv run pytest --lf                               # Run last failed tests
#
# Writing tests:
#   - Use fixtures for setup (@pytest.fixture)
#   - Use parametrize for multiple inputs (@pytest.mark.parametrize)
#   - Group tests in classes (class TestFeature)
#   - Test edge cases (empty, single element, etc.)
#   - Test error cases with pytest.raises()
