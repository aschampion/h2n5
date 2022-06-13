use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{
    Arc,
    RwLock,
};

use actix_cors::Cors;
use actix_web::middleware::Condition;
use actix_web::web::Bytes;
use actix_web::web::Data;
use actix_web::web::Query;
use actix_web::*;
use log::debug;
use lru_cache::LruCache;
use n5::filesystem::N5Filesystem;
use n5::smallvec::SmallVec;
use n5::{
    DataType,
    N5Reader,
};
use structopt::StructOpt;

mod cache;
mod encoding;
mod tiling;

use cache::N5CacheReader;
use encoding::*;
use tiling::*;

/// Trait for types that can be configured by URL query string parameters.
trait QueryConfigurable {
    fn configure(&mut self, params: &HashMap<String, String>);
}

impl QueryConfigurable for JpegParameters {
    fn configure(&mut self, params: &HashMap<String, String>) {
        if let Some(q) = params.get("q").and_then(|q| q.parse::<u8>().ok()) {
            self.quality = q;
        }
    }
}

impl QueryConfigurable for EncodingFormat {
    #[allow(unknown_lints)]
    fn configure(&mut self, params: &HashMap<String, String>) {
        if let EncodingFormat::Jpeg(ref mut p) = self {
            p.configure(params);
        }
    }
}

impl ChannelPacking {
    fn from_query(params: &HashMap<String, String>) -> Result<Self, ()> {
        if let Some(pack) = params.get("pack") {
            Self::from_str(pack)
        } else {
            Ok(ChannelPacking::default())
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum TileImageSpecError {
    #[error("Invalid value for tiling parameter: {0}")]
    InvalidValue(#[from] std::num::ParseIntError),
    #[error("Tiling request path was malformed")]
    MalformedPath,
    #[error("Unknown encoding format")]
    UnknownEncodingFormat,
    #[error("Unknown channel packing")]
    UnknownChannelPacking,
}

impl ResponseError for TileImageSpecError {
    fn error_response(&self) -> HttpResponse {
        HttpResponse::build(http::StatusCode::BAD_REQUEST).body(self.to_string())
    }
}

/// Specifies a tile to slice from a dataset and an image format to encode it.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct TileImageSpec {
    tile: TileSpec,
    format: EncodingFormat,
    packing: ChannelPacking,
}

impl FromStr for TileImageSpec {
    type Err = TileImageSpecError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        lazy_static::lazy_static! {
                static ref RE: regex::Regex = regex::Regex::new(concat!(
                r"^(?P<dataset>.*)/(?P<slicing>\d+_\d+(_\d+)?)/",
                r"(?P<tile_size>\d+_\d+)(?P<coords>(/\d+)+)\.(?P<format>.+)$"
            ))
            .expect("Impossible: regex is valid");
        }

        let caps = RE.captures(s).ok_or(TileImageSpecError::MalformedPath)?;

        let n5_dataset = caps["dataset"].into();
        let mut sd_vals = caps["slicing"].split('_').map(u32::from_str);

        let slicing_dims = SlicingDims {
            plane_dims: [sd_vals.next().unwrap()?, sd_vals.next().unwrap()?],
            channel_dim: sd_vals.next().transpose()?,
        };

        let mut ts_vals = caps["tile_size"].split('_').map(u32::from_str);

        let tile_size = TileSize {
            w: ts_vals.next().unwrap()?,
            h: ts_vals.next().unwrap()?,
        };

        let coordinates = caps["coords"]
            .split('/')
            .filter(|n| !str::is_empty(*n))
            .map(u64::from_str)
            .collect::<Result<SmallVec<_>, _>>()?;

        let format = EncodingFormat::from_str(&caps["format"])
            .map_err(|_| TileImageSpecError::UnknownEncodingFormat)?;

        let packing = ChannelPacking::default();

        Ok(TileImageSpec {
            tile: TileSpec {
                n5_dataset,
                slicing_dims,
                tile_size,
                coordinates,
            },
            format,
            packing,
        })
    }
}

#[allow(unknown_lints)]
async fn tile<N: N5Reader>(
    state: Data<AppState<N>>,
    req: HttpRequest,
    query: Query<HashMap<String, String>>,
) -> Result<HttpResponse> {
    let spec = {
        let mut spec = TileImageSpec::from_str(&req.match_info()["spec"])?;
        spec.format.configure(&query);
        spec.packing = ChannelPacking::from_query(&query)
            .map_err(|_| TileImageSpecError::UnknownChannelPacking)?;
        spec
    };

    debug!("{spec:?}");

    if spec.tile.tile_size.w > state.max_tile_size.w
        || spec.tile.tile_size.h > state.max_tile_size.h
    {
        return Ok(HttpResponse::BadRequest()
            .reason("Maximum tile size exceeded")
            .finish());
    }

    if let Some(cache) = &state.tile_cache {
        let mut cache_lock = cache.write().unwrap();
        if let Some(tile) = cache_lock.get_mut(&spec) {
            debug!("Tile cache hit");
            return Ok(HttpResponse::Ok()
                .content_type(spec.format.content_type())
                .body(tile.clone()));
        }
    }

    let n = &state.n5;
    let data_attrs = n.get_dataset_attributes(&spec.tile.n5_dataset)?;
    let mut buffer = state.tile_buffer.borrow_mut();
    buffer.clear();

    use std::ops::DerefMut;
    match *data_attrs.get_data_type() {
        DataType::UINT8 => {
            read_and_encode::<u8, _, _>(&**n, &data_attrs, &spec, buffer.deref_mut())?
        }
        DataType::UINT16 => {
            read_and_encode::<u16, _, _>(&**n, &data_attrs, &spec, buffer.deref_mut())?
        }
        DataType::UINT32 => {
            read_and_encode::<u32, _, _>(&**n, &data_attrs, &spec, buffer.deref_mut())?
        }
        DataType::UINT64 => {
            read_and_encode::<u64, _, _>(&**n, &data_attrs, &spec, buffer.deref_mut())?
        }
        DataType::FLOAT32 => {
            read_and_encode::<f32, _, _>(&**n, &data_attrs, &spec, buffer.deref_mut())?
        }
        DataType::FLOAT64 => {
            read_and_encode::<f64, _, _>(&**n, &data_attrs, &spec, buffer.deref_mut())?
        }
        _ => {
            return Ok(HttpResponse::NotImplemented()
                .reason("Data type does not have an image renderer implemented")
                .finish())
        }
    }

    // Use `as_slice().to_vec()` rather than `.clone()` to allocate the smallest
    // `Vec` possible.
    let tile: Bytes = buffer.as_slice().to_vec().into();
    let content_type = spec.format.content_type();
    if let Some(cache) = &state.tile_cache {
        cache.write().unwrap().insert(spec, tile.clone());
    }

    Ok(HttpResponse::Ok().content_type(content_type).body(tile))
}

impl ResponseError for EncodingError {
    fn error_response(&self) -> HttpResponse {
        match *self {
            Self::Io(ref e) => e.error_response(),
            Self::Image(_) => {
                HttpResponse::build(http::StatusCode::BAD_REQUEST).body(self.to_string())
            }
        }
    }
}

/// Serve N5 datasets over HTTP as tiled image stacks.
///
/// Configuration options.
#[derive(StructOpt, Debug, Clone)]
#[structopt(name = "h2n5")]
struct Options {
    /// Bind address
    #[structopt(short = "b", long = "bind-address", default_value = "127.0.0.1")]
    bind_address: std::net::IpAddr,

    /// Bind port
    #[structopt(short = "p", long = "port", default_value = "8088")]
    port: u16,

    /// Number of worker threads for handling requests.
    /// By default, the number of CPU cores is used.
    #[structopt(short = "t", long = "threads")]
    threads: Option<usize>,

    /// Allow wildcard cross-origin requests (CORS).
    ///
    /// This does not yet configure specific allowed origins,
    /// only a wildcard accept. This is most useful for
    /// development purposes. For specific CORS policies, proxy
    /// behind another HTTP server.
    #[structopt(short = "c", long = "cors")]
    cors: bool,

    /// N5 root path
    #[structopt(name = "N5_ROOT_PATH", parse(from_os_str), default_value = ".")]
    root_path: PathBuf,

    /// Maximum tile buffer preallocation
    #[structopt(long = "max-tile-prealloc", default_value = "1000000")]
    max_tile_prealloc: usize,

    /// Maximum tile size
    #[structopt(long = "max-tile-size", default_value = "4096,4096")]
    max_tile_size: TileSize,

    /// Whether to cache blocks
    #[structopt(long = "ds-block-cache")]
    ds_block_cache: bool,

    /// Cache size (in blocks) per dataset
    #[structopt(long = "ds-block-cache-size", default_value = "1024")]
    ds_block_cache_size: usize,

    /// Whether to cache tile images
    #[structopt(long = "tile-cache")]
    tile_cache: bool,

    /// Cache size (in tiles) globally
    #[structopt(long = "tile-cache-size", default_value = "1024")]
    tile_cache_size: usize,
}

type TileCache = LruCache<TileImageSpec, Bytes>;

struct AppState<N: N5Reader> {
    n5: Arc<N>,
    // `RefCell` is fine because this state is per-thread.
    tile_buffer: RefCell<Vec<u8>>,
    max_tile_size: TileSize,
    tile_cache: Option<Arc<RwLock<TileCache>>>,
}

// Must manually implement Clone because derive on generics requires them
// to be Clone even when inside an Arc (open bug in Rust 26925).
impl<N: N5Reader> Clone for AppState<N> {
    fn clone(&self) -> Self {
        AppState {
            n5: self.n5.clone(),
            tile_buffer: self.tile_buffer.clone(),
            max_tile_size: self.max_tile_size,
            tile_cache: self.tile_cache.clone(),
        }
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let opt = Options::from_args();
    let root_path = opt.root_path.to_str().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "Paths must be UTF-8")
    })?;
    let n5 = N5Filesystem::open(root_path)?;
    let max_tile_prealloc = opt.max_tile_prealloc;
    let max_tile_size = opt.max_tile_size;
    let tile_cache = if opt.tile_cache {
        Some(Arc::new(RwLock::new(TileCache::new(opt.tile_cache_size))))
    } else {
        None
    };

    if opt.ds_block_cache {
        let state = AppState {
            n5: Arc::new(N5CacheReader::wrap(n5, opt.ds_block_cache_size)),
            tile_buffer: RefCell::new(Vec::with_capacity(max_tile_prealloc)),
            max_tile_size,
            tile_cache,
        };
        run_server(opt, state).await
    } else {
        let state = AppState {
            n5: Arc::new(n5),
            tile_buffer: RefCell::new(Vec::with_capacity(max_tile_prealloc)),
            max_tile_size,
            tile_cache,
        };
        run_server(opt, state).await
    }
}

async fn run_server<N: N5Reader + Send + Sync + 'static>(
    opt: Options,
    state: AppState<N>,
) -> std::io::Result<()> {
    let cors = opt.cors;
    env_logger::init();

    let mut server = HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(state.clone()))
            .wrap(actix_web::middleware::Logger::default())
            .service(web::resource("/tile/{spec:.*}").route(web::get().to(tile::<N>)))
            .wrap(Condition::new(cors, Cors::permissive()))
    })
    .bind((opt.bind_address, opt.port))?;
    if let Some(threads) = opt.threads {
        server = server.workers(threads);
    }
    server.run().await
}

#[cfg(test)]
mod tests {
    use super::{
        EncodingFormat,
        JpegParameters,
        SlicingDims,
        TileImageSpec,
        TileSize,
    };
    use n5::smallvec::SmallVec;
    use std::str::FromStr;

    #[test]
    fn test_tile_spec_parsing() {
        let ts = TileImageSpec::from_str("my_test/dataset/0_1/512_512/3/2/1.jpg").unwrap();

        assert_eq!(ts.tile.n5_dataset, "my_test/dataset");
        assert_eq!(
            ts.tile.slicing_dims,
            SlicingDims {
                plane_dims: [0u32, 1],
                channel_dim: None,
            }
        );
        assert_eq!(ts.tile.tile_size, TileSize { w: 512, h: 512 });
        assert_eq!(ts.tile.coordinates, SmallVec::from([3u64, 2, 1]));
        assert_eq!(ts.format, EncodingFormat::Jpeg(JpegParameters::default()));
    }
}
