use ahash::RandomState;
use std::borrow::Borrow;
use std::hash::Hash;

use crate::cuckoo::{realloc_large_heap_allocated_object, Reallocator};

/// Relocate `vec`'s backing allocation through `reallocator`, in place. Trimmed
/// to a boxed slice first to drop spare capacity; the rebuilt `Vec` has
/// capacity equal to its length.
fn realloc_vec<E, R: Reallocator>(vec: &mut Vec<E>, reallocator: &mut R) {
    let mut boxed = std::mem::take(vec).into_boxed_slice();
    realloc_large_heap_allocated_object(&mut boxed, reallocator);
    *vec = boxed.into_vec();
}

#[cfg(not(feature = "linear-pq"))]
const EMPTY: u32 = u32::MAX;

/// A specialized min-heap priority queue for HeavyKeeper's top-k tracking.
///
/// Items are stored **once** in a dense `slots` array. A `Vec<u32>` binary
/// heap (of slot indices, ordered by count) provides O(1) min and O(log k)
/// insert/update.
///
/// **Lookup strategy** (compile-time):
/// - Default: open-addressing hash table for O(1) lookup.
/// - `feature = "linear-pq"`: linear scan over slots (better cache locality
///   for small K, avoids hash computation on the lookup path).
#[derive(Clone)]
pub(crate) struct TopKQueue<T> {
    slots: Vec<Slot<T>>,
    heap: Vec<u32>,
    #[cfg(not(feature = "linear-pq"))]
    table: Box<[u32]>,
    #[cfg(not(feature = "linear-pq"))]
    table_mask: u32,
    capacity: usize,
    len: usize,
    sequence: u32,
    #[cfg_attr(feature = "linear-pq", allow(dead_code))]
    hasher: RandomState,
}

#[derive(Clone)]
struct Slot<T> {
    item: T,
    count: u64,
    sequence: u32,
    heap_pos: u32,
}

impl<T: Ord + Clone + Hash + PartialEq> TopKQueue<T> {
    pub(crate) fn with_capacity_and_hasher(capacity: usize, hasher: RandomState) -> Self {
        #[cfg(not(feature = "linear-pq"))]
        let table_size = table_size_for(capacity);
        Self {
            slots: Vec::with_capacity(capacity),
            heap: Vec::with_capacity(capacity + 1),
            #[cfg(not(feature = "linear-pq"))]
            table: vec![EMPTY; table_size].into_boxed_slice(),
            #[cfg(not(feature = "linear-pq"))]
            table_mask: (table_size as u32).wrapping_sub(1),
            capacity,
            len: 0,
            sequence: 0,
            hasher,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, RandomState::new())
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Returns the heap memory (in bytes) used by this queue's containers.
    ///
    /// Computed from the allocated capacity of the slots, heap, and hash table,
    /// plus the heap each live item owns beyond its inline `size_of::<T>()`.
    /// `item_heap(t)` should return the bytes `t` points to (e.g.
    /// `String::capacity`).
    pub(crate) fn mem_bytes<F>(&self, item_heap: F) -> usize
    where
        F: Fn(&T) -> usize,
    {
        use std::mem::size_of;
        let slot_bytes = self.slots.capacity() * size_of::<Slot<T>>();
        let heap_bytes = self.heap.capacity() * size_of::<u32>();
        #[cfg(not(feature = "linear-pq"))]
        let table_bytes = self.table.len() * size_of::<u32>();
        #[cfg(feature = "linear-pq")]
        let table_bytes = 0;
        let item_bytes: usize = self.slots[..self.len].iter().map(|s| item_heap(&s.item)).sum();
        slot_bytes + heap_bytes + table_bytes + item_bytes
    }

    /// Relocate the `heap` and `slots` vectors, and the hash table, through
    /// `reallocator`. For `slots` only the outer buffer moves; any heap a `T`
    /// owns (e.g. a `Vec<u8>` key's bytes) stays put. The table `Box<[u32]>`
    /// is also relocated.
    pub(crate) fn realloc_large_heap_allocated_objects<R: Reallocator>(
        &mut self,
        reallocator: &mut R,
    ) {
        realloc_vec(&mut self.heap, reallocator);
        realloc_vec(&mut self.slots, reallocator);
        #[cfg(not(feature = "linear-pq"))]
        realloc_large_heap_allocated_object(&mut self.table, reallocator);
    }

    pub(crate) fn get<Q>(&self, item: &Q) -> Option<u64>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ToOwned<Owned = T> + ?Sized,
    {
        #[cfg(not(feature = "linear-pq"))]
        let idx = self.find_slot(item);
        #[cfg(feature = "linear-pq")]
        let idx = self.find_slot_linear(item);
        idx.map(|i| self.slots[i].count)
    }

    pub(crate) fn contains<Q>(&self, item: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        #[cfg(not(feature = "linear-pq"))]
        return self.find_slot_eq(item).is_some();
        #[cfg(feature = "linear-pq")]
        return self.find_slot_linear(item).is_some();
    }

    /// Increase an existing entry's count. Caller must guarantee the new count
    /// is >= the current count (paper Algorithm 1: heap is max(maxv, existing)).
    pub(crate) fn update_if_present<Q>(&mut self, item: &Q, count: u64) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        #[cfg(not(feature = "linear-pq"))]
        let found = self.find_slot_eq(item);
        #[cfg(feature = "linear-pq")]
        let found = self.find_slot_linear(item);

        if let Some(slot_idx) = found {
            let slot = &mut self.slots[slot_idx];
            debug_assert!(count >= slot.count, "update_if_present must not decrease");
            if count == slot.count {
                return true;
            }
            slot.count = count;
            let heap_pos = slot.heap_pos as usize;
            self.sift_down(heap_pos);
            true
        } else {
            false
        }
    }

    pub(crate) fn min_count(&self) -> u64 {
        if self.len == 0 {
            0
        } else {
            self.slots[self.heap[0] as usize].count
        }
    }

    pub(crate) fn is_full(&self) -> bool {
        self.len >= self.capacity
    }

    /// Insert or update `item` to `count`.
    ///
    /// Returns `Some(evicted)` when a previously tracked item is displaced
    /// by this call, otherwise `None`.
    pub(crate) fn upsert(&mut self, item: T, count: u64) -> Option<T> {
        // Fast path: update existing item
        #[cfg(not(feature = "linear-pq"))]
        let hash = self.hasher.hash_one(&item);
        #[cfg(not(feature = "linear-pq"))]
        let existing = self.find_slot_with_hash(&item, hash);
        #[cfg(feature = "linear-pq")]
        let existing = self.find_slot_linear(&item);

        if let Some(slot_idx) = existing {
            let slot = &mut self.slots[slot_idx];
            if count == slot.count {
                return None;
            }
            slot.count = count;
            let heap_pos = slot.heap_pos as usize;
            self.sift_down(heap_pos);
            self.sift_up(heap_pos);
            return None;
        }

        // New item: if we have space, just add it
        if self.len < self.capacity {
            if self.heap.capacity() < self.capacity + 1 {
                self.heap.reserve_exact(self.capacity + 1 - self.heap.len());
            }
            if self.slots.capacity() < self.capacity {
                self.slots
                    .reserve_exact(self.capacity - self.slots.len());
            }

            let slot_idx = self.len as u32;
            let heap_pos = self.len as u32;
            self.sequence = self.sequence.wrapping_add(1);

            self.slots.push(Slot {
                item,
                count,
                sequence: self.sequence,
                heap_pos,
            });
            self.heap.push(slot_idx);
            self.len += 1;

            #[cfg(not(feature = "linear-pq"))]
            self.table_insert(hash, slot_idx);
            self.sift_up(heap_pos as usize);
            return None;
        }

        // Queue is full — check if new count beats minimum
        if self.len > 0 {
            let min_slot_idx = self.heap[0] as usize;
            let min_count = self.slots[min_slot_idx].count;
            if count > min_count {
                #[cfg(not(feature = "linear-pq"))]
                {
                    let old_hash = self.hasher.hash_one(&self.slots[min_slot_idx].item);
                    self.table_remove(old_hash, min_slot_idx as u32);
                }

                let old_item = std::mem::replace(&mut self.slots[min_slot_idx].item, item);
                self.slots[min_slot_idx].count = count;
                self.sequence = self.sequence.wrapping_add(1);
                self.slots[min_slot_idx].sequence = self.sequence;

                #[cfg(not(feature = "linear-pq"))]
                self.table_insert(hash, min_slot_idx as u32);

                self.sift_down(0);
                return Some(old_item);
            }
        }
        None
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&T, u64)> {
        let mut items: Vec<_> = self.slots[..self.len]
            .iter()
            .map(|s| (&s.item, s.count, s.sequence))
            .collect();

        items.sort_unstable_by(|(_, c1, s1), (_, c2, s2)| match c2.cmp(c1) {
            std::cmp::Ordering::Equal => s1.cmp(s2),
            other => other,
        });

        items.into_iter().map(|(k, count, _)| (k, count))
    }

    /// Iterate items in ascending insertion-`sequence` order.
    ///
    /// Serialization uses this so restore (re-`upsert` in this order) reassigns
    /// sequences that preserve the count-tie ordering.
    pub(crate) fn iter_by_sequence(&self) -> impl Iterator<Item = (&T, u64)> {
        let mut items: Vec<_> = self.slots[..self.len]
            .iter()
            .map(|s| (&s.item, s.count, s.sequence))
            .collect();
        items.sort_unstable_by_key(|(_, _, seq)| *seq);
        items.into_iter().map(|(k, count, _)| (k, count))
    }

    // --- Linear scan lookup (feature = "linear-pq") ---

    #[cfg(feature = "linear-pq")]
    #[inline]
    fn find_slot_linear<Q>(&self, item: &Q) -> Option<usize>
    where
        T: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        for i in 0..self.len {
            if self.slots[i].item.borrow() == item {
                return Some(i);
            }
        }
        None
    }

    // --- Hash table operations (open addressing, linear probing) ---

    #[cfg(not(feature = "linear-pq"))]
    #[inline]
    fn table_bucket(&self, hash: u64) -> usize {
        (hash as u32 & self.table_mask) as usize
    }

    #[cfg(not(feature = "linear-pq"))]
    fn find_slot<Q>(&self, item: &Q) -> Option<usize>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hasher.hash_one(item);
        self.find_slot_with_hash(item, hash)
    }

    #[cfg(not(feature = "linear-pq"))]
    fn find_slot_eq<Q>(&self, item: &Q) -> Option<usize>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hasher.hash_one(item);
        self.find_slot_with_hash(item, hash)
    }

    #[cfg(not(feature = "linear-pq"))]
    #[inline]
    fn find_slot_with_hash<Q>(&self, item: &Q, hash: u64) -> Option<usize>
    where
        T: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let mut bucket = self.table_bucket(hash);
        loop {
            let slot_idx = self.table[bucket];
            if slot_idx == EMPTY {
                return None;
            }
            if self.slots[slot_idx as usize].item.borrow() == item {
                return Some(slot_idx as usize);
            }
            bucket = (bucket + 1) & (self.table_mask as usize);
        }
    }

    #[cfg(not(feature = "linear-pq"))]
    fn table_insert(&mut self, hash: u64, slot_idx: u32) {
        let mut bucket = self.table_bucket(hash);
        loop {
            if self.table[bucket] == EMPTY {
                self.table[bucket] = slot_idx;
                return;
            }
            bucket = (bucket + 1) & (self.table_mask as usize);
        }
    }

    #[cfg(not(feature = "linear-pq"))]
    fn table_remove(&mut self, hash: u64, slot_idx: u32) {
        let mut bucket = self.table_bucket(hash);
        loop {
            if self.table[bucket] == EMPTY {
                return;
            }
            if self.table[bucket] == slot_idx {
                self.table[bucket] = EMPTY;
                // Backward-shift deletion to maintain probe sequences
                let mut vacancy = bucket;
                loop {
                    bucket = (bucket + 1) & (self.table_mask as usize);
                    if self.table[bucket] == EMPTY {
                        break;
                    }
                    let natural = self.table_bucket(
                        self.hasher.hash_one(&self.slots[self.table[bucket] as usize].item),
                    );
                    // Check if this entry's natural bucket is at or before the vacancy
                    // (wrapping considered).
                    if wraps_around(natural, vacancy, bucket, self.table.len()) {
                        self.table[vacancy] = self.table[bucket];
                        self.table[bucket] = EMPTY;
                        vacancy = bucket;
                    }
                }
                return;
            }
            bucket = (bucket + 1) & (self.table_mask as usize);
        }
    }

    // --- Binary heap helpers (0-based Eytzinger layout) ---

    fn sift_up(&mut self, mut pos: usize) {
        while pos > 0 {
            let parent = (pos - 1) >> 1;
            if self.slot_count(self.heap[parent]) > self.slot_count(self.heap[pos]) {
                self.swap_nodes(parent, pos);
                pos = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut pos: usize) {
        loop {
            let mut smallest = pos;
            let left = 2 * pos + 1;
            let right = 2 * pos + 2;

            if left < self.len
                && self.slot_count(self.heap[left]) < self.slot_count(self.heap[smallest])
            {
                smallest = left;
            }
            if right < self.len
                && self.slot_count(self.heap[right]) < self.slot_count(self.heap[smallest])
            {
                smallest = right;
            }

            if smallest == pos {
                break;
            }

            self.swap_nodes(pos, smallest);
            pos = smallest;
        }
    }

    #[inline]
    fn slot_count(&self, slot_idx: u32) -> u64 {
        self.slots[slot_idx as usize].count
    }

    fn swap_nodes(&mut self, i: usize, j: usize) {
        self.heap.swap(i, j);
        self.slots[self.heap[i] as usize].heap_pos = i as u32;
        self.slots[self.heap[j] as usize].heap_pos = j as u32;
    }
}

/// Determine if `natural` is "between" `vacancy` and `current` in the circular
/// probe sequence, meaning this entry should be shifted back to fill the vacancy.
#[cfg(not(feature = "linear-pq"))]
#[inline]
fn wraps_around(natural: usize, vacancy: usize, current: usize, _len: usize) -> bool {
    if vacancy <= current {
        // No wrap: vacancy ... current (contiguous)
        natural <= vacancy || natural > current
    } else {
        // Wrapped: current ... vacancy (gap in middle)
        natural <= vacancy && natural > current
    }
}

/// Compute the hash table size (power of 2) for a given queue capacity.
/// Uses ~2x overprovisioning for a low load factor.
#[cfg(not(feature = "linear-pq"))]
fn table_size_for(capacity: usize) -> usize {
    let min = if capacity < 4 { 8 } else { capacity * 2 };
    min.next_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insertion() {
        let mut queue = TopKQueue::with_capacity(2);
        queue.upsert("a", 1);
        queue.upsert("b", 2);

        let items: Vec<_> = queue.iter().collect();
        assert_eq!(items, vec![(&"b", 2), (&"a", 1)]);
    }

    #[test]
    fn test_update_existing() {
        let mut queue = TopKQueue::with_capacity_and_hasher(2, RandomState::new());
        queue.upsert("a", 1);
        queue.upsert("b", 2);
        queue.upsert("a", 3); // Update a's count

        let items: Vec<_> = queue.iter().collect();
        assert_eq!(items, vec![(&"a", 3), (&"b", 2)]);
    }

    #[test]
    fn test_heap_cleanup() {
        let mut queue = TopKQueue::with_capacity_and_hasher(2, RandomState::new());

        // Insert initial items
        queue.upsert("a", 1);
        queue.upsert("b", 2);

        // Update 'a' multiple times
        queue.upsert("a", 3);
        queue.upsert("a", 4);
        queue.upsert("a", 5);

        // Insert new item with higher count
        queue.upsert("c", 6);

        // Check heap size vs items size
        assert_eq!(queue.heap.len(), 2, "Expected 2 items");

        let items: Vec<_> = queue.iter().collect();
        assert_eq!(items, vec![(&"c", 6), (&"a", 5)]);
    }

    #[test]
    fn test_insertion_order() {
        let mut queue = TopKQueue::with_capacity_and_hasher(3, RandomState::new());

        // Insert items with same count in specific order
        queue.upsert("a", 1);
        queue.upsert("b", 1);
        queue.upsert("c", 1);

        let items: Vec<_> = queue.iter().collect();
        assert_eq!(items, vec![(&"a", 1), (&"b", 1), (&"c", 1)]);
    }

    #[test]
    fn test_heap_consistency() {
        let mut queue = TopKQueue::with_capacity_and_hasher(2, RandomState::new());

        // Fill queue
        queue.upsert("a", 1);
        queue.upsert("b", 2);

        // Update existing item multiple times
        for i in 3..10 {
            queue.upsert("a", i);
        }

        // Try to insert new item
        queue.upsert("c", 5);

        // Verify min_count is accurate
        assert_eq!(queue.min_count(), 5);
    }

    #[test]
    fn test_capacity_overflow() {
        let mut queue = TopKQueue::with_capacity_and_hasher(2, RandomState::new());

        // Insert more items than capacity
        queue.upsert("a", 1);
        queue.upsert("b", 2);
        queue.upsert("c", 3);
        queue.upsert("d", 4);
        queue.upsert("e", 5);

        assert_eq!(queue.len(), 2, "Queue should maintain capacity");

        let items: Vec<_> = queue.iter().collect();
        assert_eq!(items, vec![(&"e", 5), (&"d", 4)]);
    }

    #[test]
    fn test_repeated_updates() {
        let mut queue = TopKQueue::with_capacity_and_hasher(2, RandomState::new());

        // Insert and update same item repeatedly
        for i in 1..100 {
            queue.upsert("a", i);
        }

        queue.upsert("b", 50);

        assert_eq!(queue.len(), 2);

        let items: Vec<_> = queue.iter().collect();
        assert_eq!(items, vec![(&"a", 99), (&"b", 50)]);
    }

    #[test]
    fn test_heap_property() {
        let mut queue = TopKQueue::with_capacity_and_hasher(10, RandomState::new());

        // Insert in reverse order to test heap maintenance
        for i in (0..=10).rev() {
            queue.upsert(format!("item{}", i), i as u64);
        }

        // Verify heap property: parent should be <= children for min-heap
        for i in 1..queue.heap.len() {
            let parent_idx = (i - 1) >> 1;
            if parent_idx > 0 {
                // Skip root's parent
                let parent_count = queue.slots[queue.heap[parent_idx] as usize].count;
                let child_count = queue.slots[queue.heap[i] as usize].count;
                assert!(
                    parent_count <= child_count,
                    "Heap property violated: parent count {} at index {} is greater than child count {} at index {}",
                    parent_count,
                    parent_idx,
                    child_count,
                    i
                );
            }
        }

        // Verify items are stored in descending order (highest counts first)
        let items: Vec<_> = queue.iter().collect();
        for i in 0..items.len() - 1 {
            assert!(
                items[i].1 >= items[i + 1].1,
                "Items not properly ordered by count: {} before {}",
                items[i].1,
                items[i + 1].1
            );
        }
    }
}
