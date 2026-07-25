//! HeavyKeeper is for finding Top-K elephant flows with high precision and low memory footprint
//!
//! This implementation is based on the paper HeavyKeeper: An Accurate Algorithm for Finding Top-k Elephant Flows
//! by Junzhi Gong, Tong Yang, Haowei Zhang, and Hao Li, Peking University; Steve Uhlig, Queen Mary, University of London;
//! Shigang Chen, University of Florida; Lorna Uden, Staffordshire University; Xiaoming Li, Peking University

// The classic and bucketed variants still store cells as fixed u64/u64; only
// the cuckoo variant is wired for the narrow-cells (u32) experiment, so gate
// the other two out of that build rather than force-narrow their `Cell` use.
#[cfg(not(feature = "narrow-cells"))]
mod heavykeeper;
#[cfg(not(feature = "narrow-cells"))]
pub use heavykeeper::{TopK, TopKDeserializeError, TopKNode};

#[cfg(not(feature = "narrow-cells"))]
mod bucketed;
#[cfg(not(feature = "narrow-cells"))]
pub use bucketed::{
    BucketedBuilderError, BucketedDeserializeError, BucketedMergeError, BucketedNode, BucketedTopK,
};

mod cuckoo;
pub use cuckoo::{
    CuckooBuilderError, CuckooDeserializeError, CuckooMergeError, CuckooNode, CuckooTopK, Reallocator,
};

mod hash_composition;
mod priority_queue;

mod serialization;
pub use serialization::DeserializeError;
