# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

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
