"""The KV-cache page allocator.

OS analogues: a block is a physical page frame, a sequence's block_table is its page table, the
free list is the free-frame list, a refcount above one is a shared page, and the prefix cache is
page deduplication backed by an LRU page cache of unreferenced but still valid blocks.

Every GPU block is in exactly one pool:
  free        refcount 0, no cached content                (free frames)
  evictable   refcount 0, still hashed and reusable         (page cache, LRU order; counts as free)
  in use      refcount >= 1, in at least one live block table
Invariant: |free| + |evictable| + |in use| == num_blocks, and every block's refcount equals the
number of live block tables containing it. Without prefix caching `evictable` stays empty and this
is the plan's `len(free) + sum(len(table)) == num_blocks`.

A separate host pool holds the blocks of swapped-out sequences.
"""

from __future__ import annotations

import hashlib
from collections import Counter, OrderedDict, deque
from collections.abc import Iterable

import numpy as np

from engine.core.sequence import Sequence


class BlockManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        enable_prefix_caching: bool = False,
        num_cpu_blocks: int = 0,
        watermark: float = 0.0,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.free_blocks: deque[int] = deque(range(num_blocks))
        self.ref_counts = [0] * num_blocks
        self.block_hash: list[bytes | None] = [None] * num_blocks
        self.hash_to_block: dict[bytes, int] = {}
        self.evictable: OrderedDict[int, None] = OrderedDict()  # oldest first
        self.num_cpu_blocks = num_cpu_blocks
        self.cpu_free: deque[int] = deque(range(num_cpu_blocks))
        self.watermark_blocks = int(watermark * num_blocks)
        self.prefix_query_tokens = 0
        self.prefix_hit_tokens = 0

    # --- accounting ------------------------------------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks) + len(self.evictable)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    def blocks_for(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    def _blocks_needed(self, seq: Sequence, num_new_tokens: int) -> int:
        return self.blocks_for(seq.num_computed_tokens + num_new_tokens) - len(seq.block_table)

    # --- prefix cache ----------------------------------------------------------------------------

    def _extend_hashes(self, seq: Sequence, num_blocks: int) -> None:
        # Chain each block's hash through its parent's so the same 16 tokens under two different
        # prefixes never collide. SHA-256 rather than hash(): a collision serves the wrong KV.
        bs = self.block_size
        hashes = seq.block_hashes
        while len(hashes) < num_blocks:
            i = len(hashes)
            tokens = np.asarray(seq.token_ids[i * bs : (i + 1) * bs], dtype=np.uint32)
            hashes.append(hashlib.sha256((hashes[-1] if hashes else b"") + tokens.tobytes()).digest())

    def cached_prefix(self, seq: Sequence) -> list[int]:
        """Blocks holding the longest cached prefix of `seq`. The block containing the last token is
        never reused: at least one token has to be recomputed to produce logits, and its K/V must
        not be written into a block other sequences may share."""
        if not self.enable_prefix_caching:
            return []
        max_blocks = (seq.num_tokens - 1) // self.block_size
        self._extend_hashes(seq, max_blocks)
        hits = []
        for h in seq.block_hashes[:max_blocks]:
            block = self.hash_to_block.get(h)
            if block is None:
                break
            hits.append(block)
        return hits

    def cache_full_blocks(self, seq: Sequence) -> None:
        """Publish blocks of `seq` that have become full. Only full blocks whose K/V are all computed
        are hashable; a partial block is still being written."""
        if not self.enable_prefix_caching:
            return
        num_full = seq.num_computed_tokens // self.block_size
        if num_full <= seq.num_registered_blocks:
            return
        self._extend_hashes(seq, num_full)
        for i in range(seq.num_registered_blocks, num_full):
            block, h = seq.block_table[i], seq.block_hashes[i]
            if self.block_hash[block] is None and h not in self.hash_to_block:
                self.block_hash[block] = h
                self.hash_to_block[h] = block
            # Otherwise the block is itself a cache hit (already published under h), or another
            # sequence published identical content first; this copy then stays private.
        seq.num_registered_blocks = num_full

    # --- allocation ------------------------------------------------------------------------------

    def can_allocate(self, seq: Sequence, num_new_tokens: int, cached: list[int], use_watermark: bool) -> bool:
        """Can `seq`, which holds no blocks, start with the `cached` prefix plus `num_new_tokens`?"""
        needed = self.blocks_for(len(cached) * self.block_size + num_new_tokens) - len(cached)
        pinned = sum(1 for b in cached if self.ref_counts[b] == 0)  # hits taken off the evictable list
        available = self.num_free_blocks - pinned - (self.watermark_blocks if use_watermark else 0)
        return needed <= available

    def allocate(self, seq: Sequence, num_new_tokens: int, cached: list[int]) -> None:
        assert not seq.block_table, f"seq {seq.seq_id} already holds blocks"
        for block in cached:
            if self.ref_counts[block] == 0:
                del self.evictable[block]
            self.ref_counts[block] += 1
        seq.block_table = list(cached)
        seq.num_computed_tokens = len(cached) * self.block_size
        seq.num_cached_tokens = seq.num_computed_tokens
        seq.num_registered_blocks = len(cached)
        if self.enable_prefix_caching:
            self.prefix_query_tokens += seq.num_tokens
            self.prefix_hit_tokens += seq.num_computed_tokens
        self._grow(seq, num_new_tokens)

    def can_append(self, seq: Sequence, num_new_tokens: int) -> bool:
        return self._blocks_needed(seq, num_new_tokens) <= self.num_free_blocks

    def append_slots(self, seq: Sequence, num_new_tokens: int) -> None:
        self._grow(seq, num_new_tokens)

    def free(self, seq: Sequence) -> None:
        # Tail first: when a sequence's blocks land on the LRU list, its tail is evicted before
        # its prefix, which is the part other requests are likely to share.
        for block in reversed(seq.block_table):
            self._release(block)
        seq.block_table = []
        seq.num_registered_blocks = 0

    def _grow(self, seq: Sequence, num_new_tokens: int) -> None:
        for _ in range(self._blocks_needed(seq, num_new_tokens)):
            seq.block_table.append(self._take_free_block())

    def _take_free_block(self) -> int:
        if self.free_blocks:
            block = self.free_blocks.popleft()
        else:
            block, _ = self.evictable.popitem(last=False)  # page replacement: evict the LRU cached block
            del self.hash_to_block[self.block_hash[block]]
            self.block_hash[block] = None
        self.ref_counts[block] = 1
        return block

    def _release(self, block: int) -> None:
        self.ref_counts[block] -= 1
        if self.ref_counts[block] == 0:
            if self.block_hash[block] is not None:
                self.evictable[block] = None
            else:
                self.free_blocks.append(block)

    # --- swapping --------------------------------------------------------------------------------

    def can_swap_out(self, seq: Sequence) -> bool:
        return len(seq.block_table) <= len(self.cpu_free)

    def swap_out(self, seq: Sequence) -> list[tuple[int, int]]:
        """Move seq's KV to host blocks. Returns (gpu_block, cpu_block) copies for the runner, which
        performs them before this step's forward pass, so the freed GPU blocks can be reused."""
        pairs = []
        for block in seq.block_table:
            cpu_block = self.cpu_free.popleft()
            pairs.append((block, cpu_block))
            seq.cpu_block_table.append(cpu_block)
        self.free(seq)
        return pairs

    def can_swap_in(self, seq: Sequence, num_new_tokens: int, use_watermark: bool) -> bool:
        needed = self.blocks_for(seq.num_computed_tokens + num_new_tokens)
        return needed <= self.num_free_blocks - (self.watermark_blocks if use_watermark else 0)

    def swap_in(self, seq: Sequence) -> list[tuple[int, int]]:
        """Bring seq's KV back into fresh GPU blocks. Returns (cpu_block, gpu_block) copies. Never
        combined with swap_out in one step (the scheduler guarantees it), so a host block released
        here cannot be overwritten before it is read."""
        assert not seq.block_table
        pairs = []
        for cpu_block in seq.cpu_block_table:
            block = self._take_free_block()
            pairs.append((cpu_block, block))
            seq.block_table.append(block)
            self.cpu_free.append(cpu_block)
        seq.cpu_block_table = []
        seq.num_registered_blocks = 0  # re-publish on the next step; duplicates are skipped
        return pairs

    def free_cpu(self, seq: Sequence) -> None:
        self.cpu_free.extend(seq.cpu_block_table)
        seq.cpu_block_table = []

    # --- invariants ------------------------------------------------------------------------------

    def check_invariants(self, gpu_seqs: Iterable[Sequence], cpu_seqs: Iterable[Sequence] = ()) -> None:
        """Assert the pool accounting. `gpu_seqs` must be every sequence holding GPU blocks,
        `cpu_seqs` every sequence holding host blocks. Call between steps."""
        counts: Counter[int] = Counter()
        for seq in gpu_seqs:
            table = seq.block_table
            assert len(set(table)) == len(table), f"seq {seq.seq_id} maps a block twice: {table}"
            assert len(table) == self.blocks_for(seq.num_computed_tokens), (
                f"seq {seq.seq_id}: {len(table)} blocks for {seq.num_computed_tokens} computed tokens"
            )
            counts.update(table)
        for block in range(self.num_blocks):
            assert self.ref_counts[block] == counts[block], (
                f"block {block}: refcount {self.ref_counts[block]} but held by {counts[block]} block tables"
            )
        free, evictable, in_use = set(self.free_blocks), set(self.evictable), set(counts)
        assert len(free) == len(self.free_blocks), "free list holds a block twice"
        assert not (free & evictable or free & in_use or evictable & in_use), "a block is in two pools"
        total = len(free) + len(evictable) + len(in_use)
        assert total == self.num_blocks, (
            f"leak: {len(free)} free + {len(evictable)} evictable + {len(in_use)} in use != {self.num_blocks}"
        )
        for block in free:
            assert self.block_hash[block] is None, f"free block {block} still has a hash"
        for block in evictable:
            h = self.block_hash[block]
            assert h is not None and self.hash_to_block.get(h) == block, f"evictable block {block} is not cached"
        for h, block in self.hash_to_block.items():
            assert self.block_hash[block] == h, f"hash map and block {block} disagree"

        cpu_counts = Counter(c for seq in cpu_seqs for c in seq.cpu_block_table)
        assert all(n == 1 for n in cpu_counts.values()), "a host block is held twice"
        cpu_free = set(self.cpu_free)
        assert len(cpu_free) == len(self.cpu_free), "host free list holds a block twice"
        assert not cpu_free & set(cpu_counts), "a host block is both free and held"
        assert len(cpu_free) + len(cpu_counts) == self.num_cpu_blocks, (
            f"host leak: {len(cpu_free)} free + {len(cpu_counts)} held != {self.num_cpu_blocks}"
        )
