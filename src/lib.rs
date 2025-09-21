#![allow(clippy::missing_safety_doc)]

/// Original implementations of the algorithms.
pub(crate) mod original;
pub mod prelude;
pub mod traits;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64;
