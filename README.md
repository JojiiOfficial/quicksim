# Quicksim
Quick-sim(d) provides several <b>SIMD-accelerated, drop-in replacements</b> for common algorithms.<br>
It's designed to make speeding up trivial parts of your code as easy as possible - <b>no manual SIMD programming required</b>.
AVX features are <b>automatically detected at runtime</b> and enabled whenever possible. <br>
Quick-sim(d) is not intended as a replacement for [Portable SIMD](https://github.com/rust-lang/portable-simd). Instead, it aims to <b>completely hide vectorization logic</b> from the API, keeping usage simple and ergonomic. <br>
The project is currently in <b>alpha state</b>, and does <b>not yet provide vectorized implementations</b> for all vectorizable functions in the Rust standard library. <br>
<br>
The implementations in this crate generally achieve significant speedups compared to naive versions. However, there are some trade-offs. For more information, see the [Limitations](#Limitations) section.

`Note: Iterators are usually fast enough for most cases. Use this crate carefully, and make sure to identify critical hot-paths before replacing all of your implementations with the accelerated functions provided by this crate.`

# Add to your project

```
quicksim = "0.1"
```

# Example
```rust
use quicksim::prelude::*;

fn main() {
    // Note: SIMD implementations only start to take effect with arrays of 32 elements or more.
    let array: Vec<u32> = vec![0, 1, 2, 3, 4];

    assert_eq!(array.max_simd(), array.iter().max().copied());
    assert_eq!(array.min_simd(), array.iter().min().copied());
    assert_eq!(array.find_simd(4), Some(4));
    assert!(array.contains_simd(4));
}
```


# Limitations
The SIMD implementation becomes effective for arrays with more than 32 items. This means that if your array length is below 32 more than 50% of the time, using the `*_simd()` functions of this crate will generally be slower on average.
Additionally, if no optimized implementation exists for a specific architecture, the SIMD version will always be slower than the regular implementation due to the overhead of checking for SIMD support and falling back to the default version.

# Attribution
Many algorithms implemented here are based on those from [Algorithmica](https://en.algorithmica.org/hpc/), which helped me to get deeper into SIMD and implement some of the algorithms in this crate.
Check this book out if you're interested in SIMD programming yourself, or if you want to learn more about computers, algorithms and low-level optimizations.

# Benchmarks

The SIMD implementation becomes effective for arrays with <b>more than 32 items</b>. For smaller arrays (32 items or fewer), the SIMD versions are actually <b>slower</b> than the standard implementations. <br>
Please note that the benchmark results <b>may contain some noise</b>, so they shouldn't be treated as 100% accurate.<br>
All measurements were taken on a <b>Ryzen 5 5600x</b> CPU with <b>32 GB of RAM</b>.

## &[u32]
### 🔍 Find - avg
|    |   size |     simd |     iter |   speedup |
|----|--------|----------|----------|-----------|
|  0 |     16 |   2.8042 |   2.367  |  0.844091 |
|  1 |     32 |   2.1073 |   4.7695 |  2.26332  |
|  2 |     64 |   2.5788 |   8.4581 |  3.27986  |
|  3 |    150 |   3.5278 |  22.757  |  6.45076  |
|  4 |    530 |   9.2504 |  67.652  |  7.31341  |
|  5 |   1028 |  16.543  | 126.14   |  7.62498  |
|  6 |   5010 |  85.771  | 595.31   |  6.94069  |
|  7 |   8000 | 132.41   | 948.74   |  7.16517  |

### 🔍 Find - worst
|    |   size |     simd |      iter |   speedup |
|----|--------|----------|-----------|-----------|
|  0 |     16 |   5.1783 |    4.1654 |  0.804395 |
|  1 |     32 |   2.3703 |    8.2243 |  3.46973  |
|  2 |     64 |   3.0473 |   15.68   |  5.14554  |
|  3 |    150 |  11.059  |   40.107  |  3.62664  |
|  4 |    530 |  20.414  |  129.42   |  6.33977  |
|  5 |   1028 |  36.596  |  245.98   |  6.7215   |
|  6 |   5010 | 153.56   | 1178.3    |  7.67322  |
|  7 |   8000 | 243.77   | 1880.8    |  7.71547  |

### Count
|    |   size |     simd |      iter |   speedup |
|----|-------|----------|-----------|-----------|
|  0 |     16 |   2.3437 |    2.6018 |   1.11013 |
|  1 |     32 |   2.8135 |    4.6383 |   1.64859 |
|  2 |     64 |   3.0854 |    8.9749 |   2.90883 |
|  3 |    150 |   7.3019 |   20.38   |   2.79105 |
|  4 |    530 |  12.712  |   73.599  |   5.78973 |
|  5 |   1028 |  18.706  |  137.6    |   7.35593 |
|  6 |   5010 |  83.04   |  658.32   |   7.92775 |
|  7 |   8000 | 126.9    | 1061.6    |   8.36564 |

### Min
|    |   size |     simd |      iter |   speedup |
|----|--------|----------|-----------|----------|
|  0 |     16 |   6.8805 |    4.4347 |  0.644532 |
|  1 |     32 |   2.5866 |    7.578  |  2.92971  |
|  2 |     64 |   3.2832 |   15.214  |  4.63389  |
|  3 |    150 |  10.318  |   35.093  |  3.40114  |
|  4 |    530 |  13.826  |  116.56   |  8.43049  |
|  5 |   1028 |  18.393  |  270.9    | 14.7284   |
|  6 |   5010 |  80.394  | 1195.6    | 14.8718   |
|  7 |   8000 | 126.19   | 1885.2    | 14.9394   |

### Max
|    |   size |     simd |      iter |   speedup |
|----|--------|----------|-----------|-----------|
|  0 |     16 |   5.3713 |    4.9116 |  0.914416 |
|  1 |     32 |   2.5786 |    9.9717 |  3.8671   |
|  2 |     64 |   3.0447 |   24.564  |  8.06779  |
|  3 |    150 |  12.016  |   65.061  |  5.41453  |
|  4 |    530 |  14.317  |  244.71   | 17.0923   |
|  5 |   1028 |  18.343  |  479.27   | 26.1282   |
|  6 |   5010 |  80.861  | 2337      | 28.9014   |
|  7 |   8000 | 126.19   | 3734      | 29.5903   |

### Contains - avg
|    |   size |     simd |     iter |   speedup |
|----|--------|----------|----------|-----------|
|  0 |     16 |   2.5754 |   2.5422 |  0.987109 |
|  1 |     32 |   1.8953 |   4.401  |  2.32206  |
|  2 |     64 |   2.3444 |   8.6579 |  3.69301  |
|  3 |    150 |   3.4732 |  18.912  |  5.44512  |
|  4 |    530 |   8.1109 |  68.4    |  8.4331   |
|  5 |   1028 |  17.607  | 127.71   |  7.25337  |
|  6 |   5010 |  88.245  | 590.65   |  6.6933   |
|  7 |   8000 | 129.19   | 941.28   |  7.28601  |

### Contains - worst
|    |   size |     simd |      iter |   speedup |
|----|--------|----------|-----------|-----------|
|  0 |     16 |   4.8998 |    4.9446 |   1.00914 |
|  1 |     32 |   2.1081 |    8.6596 |   4.10777 |
|  2 |     64 |   2.5934 |   20.38   |   7.85841 |
|  3 |    150 |  12.114  |   40.873  |   3.37403 |
|  4 |    530 |  21.691  |  130.17   |   6.00111 |
|  5 |   1028 |  31.116  |  249.47   |   8.01742 |
|  6 |   5010 | 154.94   | 1192.1    |   7.69395 |
|  7 |   8000 | 245.39   | 1874.9    |   7.64049 |

## &[f32]
### Min
|    |   size |     simd |      iter |   speedup |
|----|--------|----------|-----------|-----------|
|  0 |     16 |   3.46   |    3.2625 |  0.942919 |
|  1 |     32 |   2.3321 |    5.1986 |  2.22915  |
|  2 |     64 |   2.8477 |   10.93   |  3.83819  |
|  3 |    150 |   6.8676 |   27.82   |  4.05091  |
|  4 |    530 |  14.967  |  118.27   |  7.90205  |
|  5 |   1028 |  21.059  |  234.49   | 11.1349   |
|  6 |   5010 |  79.785  | 1177.4    | 14.7572   |
|  7 |   8000 | 128.59   | 1859.9    | 14.4638   |

### Max
|    |   size |     simd |      iter |   speedup |
|----|--------|----------|-----------|-----------|
|  0 |     16 |   3.6937 |    3.2061 |  0.867991 |
|  1 |     32 |   2.3294 |    5.0832 |  2.18219  |
|  2 |     64 |   3.299  |   10.929  |  3.31282  |
|  3 |    150 |   6.7782 |   28.02   |  4.13384  |
|  4 |    530 |  14.808  |  118.77   |  8.02066  |
|  5 |   1028 |  25.172  |  234.02   |  9.29684  |
|  6 |   5010 | 120.41   | 1157.3    |  9.61133  |
|  7 |   8000 | 190.97   | 1861.4    |  9.74708  |
