# H2N5 [![Build Status](https://github.com/aschampion/h2n5/actions/workflows/ci.yml/badge.svg)](https://github.com/aschampion/h2n5/actions/workflows/ci.yml/)

H2N5 is:
- ~~An Australian strain of avian flu~~
- ~~[(E)-Hydrazinylidenehydrazinylidene]azanide~~
- HTTP 2 N5: A simple program to serve N5 datasets over HTTP as tiled image stacks

## Minimum supported Rust version (MSRV)

Stable 1.56

## Quick start

Installation options:
- Compile and install using cargo: `cargo install h2n5`
- Download a precompiled binary from [the latest GitHub releases](https://github.com/aschampion/h2n5/releases/latest)
- Install a precompiled binary using [cargo-binstall](https://github.com/ryankurte/cargo-binstall): `cargo binstall h2n5`

```
h2n5 path/to/my.n5
curl http://127.0.0.1:8088/tile/group/dataset/0_1/512_256/1/2/3.jpg?q=80
```

Tile URLs are constructed as:

```
http://[bind_address]:[port]/tile/[n5_dataset]/[slicing_dims]/[tile_size]/[coordinates].[format]?[query parameters]
```

For example, the URL:

```
http://127.0.0.1:8088/tile/group/dataset/0_1/512_256/1/2/3.jpg?q=80
```

Will slice a 512 px by 256 px tile from the `group/dataset` dataset along axes `0` (as tile X) and `1` (as tile Y), respectively. The returned tile will start at voxel coordinates [1, 2, 3], be encoded as JPEG, with a quality of 80.

For more options, see the command line help:

```
h2n5 -h
```

## Notes

- PNG (`png`) and JPEG (`jpg`|`jpeg`) encoding formats are supported.
- Currently, only grayscale tiles are returned (by slicing remaining dimensions as singletons) or grayscale + alpha or RGBA by packing these scalar values across channels using the `pack` query parameter. Slicing a third dimension for RGB(A) channels (e.g., `slicing_dims` as `0_1_4`) will be supported, and is currently parsed correctly, but not yet implemented for tile encoding.

### Supported data type, channel packing, and encoding formats

|        | gray (default) | graya      | rgba       |
|--------|----------------|------------|------------|
| UINT8  | JPEG PNG       |            |            |
| UINT16 | PNG            | JPEG PNG   |            |
| UINT32 |                | PNG        | JPEG PNG   |
| UINT64 |                |            | PNG        |
| FLOAT32|                | PNG        | JPEG PNG   |
| FLOAT64|                |            | PNG        |

### Logging

Logs are output to stderr using standard Rust `log` and `env_logger` in the `actix_web` scope. By default only major errors at the `WARN` level or above are shown. For example, to show all requests:

```
RUST_LOG=actix_web=info h2n5
```

For more information, see the [`env_logger`](https://docs.rs/env_logger) and [`actix_web` logging middleware](https://actix.rs/docs/errors/) documentation.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
