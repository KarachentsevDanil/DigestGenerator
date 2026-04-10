from __future__ import annotations

import pickle
import re

from datasketch import MinHash, MinHashLSH

from src.db.models import Message


class MinHashIndex:
    """Thin wrapper around datasketch MinHashLSH for near-duplicate detection."""

    def __init__(self, threshold: float = 0.7, num_perm: int = 128):
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self._keys: set[str] = set()

    def insert(self, key: str, minhash: MinHash) -> None:
        """Insert a MinHash with a unique key (message ID)."""
        if key not in self._keys:
            self.lsh.insert(key, minhash)
            self._keys.add(key)

    def query(self, minhash: MinHash) -> list[str]:
        """Return keys of similar items above threshold."""
        return self.lsh.query(minhash)

    def build_from_messages(self, messages: list[Message]) -> None:
        """Bulk load from DB messages (deserialize minhash_signature)."""
        for msg in messages:
            if msg.minhash_signature:
                mh = pickle.loads(msg.minhash_signature)
                self.insert(str(msg.id), mh)

    @staticmethod
    def create_minhash(text: str, num_perm: int = 128) -> MinHash:
        """Create MinHash from text using word 3-gram shingles."""
        mh = MinHash(num_perm=num_perm)
        words = re.findall(r"\w+", text.lower())
        n = 3
        if len(words) < n:
            shingles = [" ".join(words)] if words else [""]
        else:
            shingles = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
        for shingle in shingles:
            mh.update(shingle.encode("utf-8"))
        return mh

    @staticmethod
    def deserialize(data: bytes) -> MinHash:
        """Deserialize a stored MinHash."""
        return pickle.loads(data)
