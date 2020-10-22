# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [0.1.7] - 2020-10-22
### Fixed
- Fixed a soundness error in a dependency that caused a panic on Rust 1.48 and
  greater (see
  [this issue](https://github.com/contain-rs/linked-hash-map/pull/100)).
- Reduced allocation frequency and size for encoded tiles.

## [0.1.6] - 2020-07-18
### Fixed
- Fixed a panic when a block's dataset-level bounds overlapped a tile request
  but the block's actual bounds did not.

## [0.1.5] - 2020-07-13
### Added
- An optional LRU memory cache for tiles can be enabled with the `--tile-cache`
  option. The size of the cache (in tiles) globally can be configured with the
  `--tile-cache-size` option. HTTP server caching should be preferred for
  production deployments, but this option is useful for temporary cases.

### Fixed
- Fixes from dependencies include a bug that caused n5 to corrupt the version
  attribute when opening datasets and extremely rare cases where PNG tiles
  would be corrupted during compression.

## [0.1.4] - 2019-11-15
### Added
- An optional LRU memory cache for blocks can be enabled with the
  `--ds-block-cache` option. The size of the cache (in blocks) per-dataset can
  be configured with the `--ds-block-cache-size` option.
- Scalar value bytes can be packed into channels by adding a `pack` query
  parameter to request URLs. For example: `?pack=rgba`.
- `FLOAT32` and `FLOAT64` datasets are now supported via channel packing.
- The maximum size (in bytes) that will be preallocated for a tile before
  decoding can be configured with `--max-tile-prealloc`.
- Logging is now enabled via `env_logger`.

### Changed
- Actix-web and many other dependencies underwent major update that include
  security issues.
- Cumulative optimizations can improve request response times by 40%.

### Fixed
- A bug that could cause a panic on some platforms and environments has been
  fixed.
