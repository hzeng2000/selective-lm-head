from selective_lm_head.grammar.trie_allowed_set import TokenTrie, TrieAllowedSetProvider


def test_trie_allowed_next_and_accepts():
    trie = TokenTrie(eos_token_id=99)
    trie.insert([1, 2, 3], value="abc")
    trie.insert([1, 4], value="ad")

    assert trie.allowed_next([]) == [1]
    assert trie.allowed_next([1]) == [2, 4]
    assert trie.allowed_next([1, 2, 3]) == [99]
    assert trie.accepts([1, 2, 3])
    assert not trie.accepts([1, 2])


def test_trie_provider_state():
    trie = TokenTrie()
    trie.insert([7, 8])
    provider = TrieAllowedSetProvider(trie)
    assert provider.allowed_token_ids() == [7]
    provider.accept_token(7)
    assert provider.allowed_token_ids() == [8]

