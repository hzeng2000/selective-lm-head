"""Direct allowed-id provider for fixed token tries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class TrieNode:
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    terminal: bool = False
    values: list[str] = field(default_factory=list)


class TokenTrie:
    def __init__(self, eos_token_id: int | None = None):
        self.root = TrieNode()
        self.eos_token_id = eos_token_id

    def insert(self, token_ids: Iterable[int], value: str | None = None) -> None:
        node = self.root
        for token_id in token_ids:
            node = node.children.setdefault(int(token_id), TrieNode())
        node.terminal = True
        if value is not None:
            node.values.append(value)

    def node_for_prefix(self, prefix: Iterable[int]) -> TrieNode | None:
        node = self.root
        for token_id in prefix:
            node = node.children.get(int(token_id))
            if node is None:
                return None
        return node

    def allowed_next(self, prefix: Iterable[int]) -> list[int]:
        node = self.node_for_prefix(prefix)
        if node is None:
            return []
        allowed = sorted(node.children)
        if node.terminal and self.eos_token_id is not None:
            allowed.append(self.eos_token_id)
        return allowed

    def accepts(self, token_ids: Iterable[int]) -> bool:
        node = self.node_for_prefix(token_ids)
        return bool(node and node.terminal)


class TrieAllowedSetProvider:
    """Stateful provider for one decode sequence over a TokenTrie."""

    name = "direct_trie"

    def __init__(self, trie: TokenTrie):
        self.trie = trie
        self.prefix: list[int] = []

    def allowed_token_ids(self) -> list[int]:
        return self.trie.allowed_next(self.prefix)

    def accept_token(self, token_id: int) -> None:
        self.prefix.append(int(token_id))

    def is_terminated(self) -> bool:
        return self.trie.accepts(self.prefix)

    def reset(self) -> None:
        self.prefix.clear()

