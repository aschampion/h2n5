use std::collections::HashMap;
use std::io::{
    Write,
};
use std::num::ParseIntError;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use actix_web::*;
use actix_cors::Cors;
use actix_web::web::Data;
use actix_web::web::Query;
use n5::{
    DatasetAttributes,
    DataType,
    N5Reader,
    ReinitDataBlock,
    ReflectedType,
    ReadableDataBlock,
};
use n5::filesystem::{
    N5Filesystem,
};
use n5::ndarray::prelude::*;
use n5::smallvec::{SmallVec, smallvec};
use structopt::StructOpt;


mod cache;

use cache::N5CacheReader;


#[derive(Debug, PartialEq)]
struct SlicingDims {
    plane_dims: [u32; 2],
    channel_dim: Option<u32>,
}

/// Trait for types that can be configured by URL query string parameters.
trait QueryConfigurable {
    fn configure(&mut self, params: &HashMap<String, String>);
}

#[derive(Debug, PartialEq)]
struct JpegParameters {
    quality: u8,
}

impl Default for JpegParameters {
    fn default() -> JpegParameters {
        JpegParameters {
            quality: 100,
        }
    }
}

impl QueryConfigurable for JpegParameters {
    fn configure(&mut self, params: &HashMap<String, String>) {
        if let Some(q) = params.get("q").and_then(|q| q.parse::<u8>().ok()) {
            self.quality = q;
        }
    }
}

#[derive(Debug, PartialEq)]
enum EncodingFormat {
    Jpeg(JpegParameters),
    Png,
}

impl EncodingFormat {
    fn encode<W: Write>(
        &self,
        writer: &mut W,
        bytes: &[u8],
        tile_size: &TileSize,
        image_color_type: image::ColorType,
    ) -> Result<(), std::io::Error> {
        match *self {
            EncodingFormat::Jpeg(ref p) => {
                let mut encoder = image::jpeg::JPEGEncoder::new_with_quality(writer, p.quality);
                encoder.encode(
                    bytes,
                    tile_size.w,
                    tile_size.h,
                    image_color_type)
            },
            EncodingFormat::Png => {
                let encoder = image::png::PNGEncoder::new(writer);
                encoder.encode(
                    bytes,
                    tile_size.w,
                    tile_size.h,
                    image_color_type)
            }
        }
    }

    fn content_type(&self) -> &'static str {
        match *self {
            EncodingFormat::Jpeg(_) => "image/jpeg",
            EncodingFormat::Png => "image/png",
        }
    }
}

impl FromStr for EncodingFormat {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "jpg" | "jpeg" => Ok(EncodingFormat::Jpeg(JpegParameters::default())),
            "png" => Ok(EncodingFormat::Png),
            _ => Err(()),
        }
    }
}

impl QueryConfigurable for EncodingFormat {
    #[allow(unknown_lints)]
    fn configure(&mut self, params: &HashMap<String, String>) {
        match *self {
            EncodingFormat::Jpeg(ref mut p) => p.configure(params),
            _ => (),
        }
    }
}

#[derive(Debug, PartialEq)]
enum ChannelPacking {
    Gray,
    GrayA,
    RGBA,
}

impl ChannelPacking {
    fn from_query(params: &HashMap<String, String>) -> Result<Self, ()> {
        if let Some(pack) = params.get("pack") {
            Self::from_str(pack)
        } else {
            Ok(ChannelPacking::default())
        }
    }

    fn num_channels(&self) -> u8 {
        match self {
            ChannelPacking::Gray => 1,
            ChannelPacking::GrayA => 2,
            ChannelPacking::RGBA => 4,
        }
    }
}

impl Default for ChannelPacking {
    fn default() -> Self {
        ChannelPacking::Gray
    }
}

impl FromStr for ChannelPacking {
    type Err = ();  // TODO

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "gray" => Ok(ChannelPacking::Gray),
            "graya" => Ok(ChannelPacking::GrayA),
            "rgba" => Ok(ChannelPacking::RGBA),
            _ => Err(()),
        }
    }
}

#[derive(Debug)]
enum TileSpecError {
    InvalidValue(std::num::ParseIntError),
    MalformedPath,
    UnknownEncodingFormat,
    UnknownChannelPacking,
}

impl std::fmt::Display for TileSpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::error::Error;
        match *self {
            TileSpecError::InvalidValue(ref e) => write!(f, "{}: {}", self.description(), e),
            _ => write!(f, "{}", self.description()),
        }
    }
}

impl std::error::Error for TileSpecError {
    fn description(&self) -> &str {
        match *self {
            TileSpecError::InvalidValue(_) => "Invalid value for tiling parameter",
            TileSpecError::MalformedPath => "Tiling request path was malformed",
            TileSpecError::UnknownEncodingFormat => "Unknown encoding format",
            TileSpecError::UnknownChannelPacking => "Unknown channel packing",
        }
    }
}

impl From<std::num::ParseIntError> for TileSpecError {
    fn from(e: std::num::ParseIntError) -> TileSpecError {
        TileSpecError::InvalidValue(e)
    }
}

impl ResponseError for TileSpecError {
    fn error_response(&self) -> HttpResponse {
        match *self {
            _ => HttpResponse::build(http::StatusCode::BAD_REQUEST)
                    .body(self.to_string()),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct TileSize {
    w: u32,
    h: u32,
}

impl FromStr for TileSize {
    type Err = ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let coords: SmallVec<[&str; 3]> = s.split(',').collect();

        let w = coords[0].parse::<u32>()?;
        let h = coords[1].parse::<u32>()?;

        Ok(TileSize {w, h})
    }
}

impl std::fmt::Display for TileSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{}", self.w, self.h)
    }
}

#[derive(Debug)]
struct TileSpec {
    n5_dataset: String,
    slicing_dims: SlicingDims,
    tile_size: TileSize,
    coordinates: SmallVec<[u64; 6]>,
    format: EncodingFormat,
    packing: ChannelPacking,
}

impl FromStr for TileSpec {

    type Err = TileSpecError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let re = regex::Regex::new(concat!(
                r"^(?P<dataset>.*)/(?P<slicing>\d+_\d+(_\d+)?)/",
                r"(?P<tile_size>\d+_\d+)(?P<coords>(/\d+)+)\.(?P<format>.+)$"))
            .expect("Impossible: regex is valid");
        let caps = re.captures(s).ok_or(TileSpecError::MalformedPath)?;

        let n5_dataset = caps["dataset"].into();
        let mut sd_vals = caps["slicing"].split('_').map(u32::from_str);

        let slicing_dims = SlicingDims {
            plane_dims: [
                sd_vals.next().unwrap()?,
                sd_vals.next().unwrap()?,
            ],
            channel_dim: sd_vals.next().transpose()?,
        };

        let mut ts_vals = caps["tile_size"].split('_').map(u32::from_str);

        let tile_size = TileSize {
            w: ts_vals.next().unwrap()?,
            h: ts_vals.next().unwrap()?,
        };

        let coordinates = caps["coords"].split('/')
            .filter(|n| !str::is_empty(*n))
            .map(u64::from_str)
            .collect::<Result<SmallVec<_>, _>>()?;

        let format = EncodingFormat::from_str(&caps["format"])
            .map_err(|_| TileSpecError::UnknownEncodingFormat)?;

        let packing = ChannelPacking::default();

        Ok(TileSpec {
            n5_dataset,
            slicing_dims,
            tile_size,
            coordinates,
            format,
            packing,
        })
    }
}

#[allow(unknown_lints)]
fn tile(
    state: Data<AppState>,
    req: HttpRequest,
    query: Query<HashMap<String, String>>,
) -> Result<HttpResponse> {
    let spec = {
        let mut spec = TileSpec::from_str(&req.match_info()["spec"])?;
        spec.format.configure(&query);
        spec.packing = ChannelPacking::from_query(&query)
            .map_err(|_| TileSpecError::UnknownChannelPacking)?;
        spec
    };

    if spec.tile_size.w > state.max_tile_size.w ||
       spec.tile_size.h > state.max_tile_size.h {
        return Ok(HttpResponse::BadRequest()
            .reason("Maximum tile size exceeded")
            .finish());
    }

    let n = &state.n5cache;
    let data_attrs = n.get_dataset_attributes(&spec.n5_dataset)?;
    // Allocate a buffer large enough for the uncompressed tile, as the
    // compressed size will be less with high probability.
    let buffer_size = spec.tile_size.w as usize * spec.tile_size.h as usize
        * spec.slicing_dims.channel_dim.map(|d| data_attrs.get_dimensions()[d as usize] as usize).unwrap_or(1)
        * data_attrs.get_data_type().size_of();
    let buffer_size = std::cmp::min(buffer_size, state.max_tile_prealloc);
    let mut tile_buffer: Vec<u8> = Vec::with_capacity(buffer_size);

    match *data_attrs.get_data_type() {
        DataType::UINT8 => read_and_encode::<u8, _, _>(&**n, &data_attrs, &spec, &mut tile_buffer)?,
        DataType::UINT16 => read_and_encode::<u16, _, _>(&**n, &data_attrs, &spec, &mut tile_buffer)?,
        DataType::UINT32 => read_and_encode::<u32, _, _>(&**n, &data_attrs, &spec, &mut tile_buffer)?,
        DataType::UINT64 => read_and_encode::<u64, _, _>(&**n, &data_attrs, &spec, &mut tile_buffer)?,
        DataType::FLOAT32 => read_and_encode::<f32, _, _>(&**n, &data_attrs, &spec, &mut tile_buffer)?,
        DataType::FLOAT64 => read_and_encode::<f64, _, _>(&**n, &data_attrs, &spec, &mut tile_buffer)?,
        _ => return Ok(HttpResponse::NotImplemented()
            .reason("Data type does not have an image renderer implemented")
            .finish()),
    }
    Ok(HttpResponse::Ok()
        .content_type(spec.format.content_type())
        .body(tile_buffer))
}

// TODO: Single channel only.
fn read_and_encode<T, N: N5Reader, W: Write>(
    n: &N,
    data_attrs: &DatasetAttributes,
    spec: &TileSpec,
    writer: &mut W,
) -> Result<(), std::io::Error>
where n5::VecDataBlock<T>: n5::DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock,
      T: ReflectedType + num_traits::identities::Zero {

    // Express the spec tile as an N-dim bounding box.
    let mut size = smallvec![1u64; data_attrs.get_dimensions().len()];
    size[spec.slicing_dims.plane_dims[0] as usize] = u64::from(spec.tile_size.w);
    size[spec.slicing_dims.plane_dims[1] as usize] = u64::from(spec.tile_size.h);
    if let Some(dim) = spec.slicing_dims.channel_dim {
        size[dim as usize] = data_attrs.get_dimensions()[dim as usize];
    }
    let bbox = BoundingBox::new(spec.coordinates.clone(), size);

    // Read the N-dim slab of blocks containing the tile from N5.
    let slab = n.read_ndarray::<T>(
        &spec.n5_dataset,
        data_attrs,
        &bbox)?;

    let image_color_type = match spec.slicing_dims.channel_dim {
        Some(_dim) => {
            // TODO: match RGB/RGBA based on dimensions of dim.
            // Permute slab so that channels dimension is at end.
            unimplemented!()
        },
        None => {
            let bits_per_channel = 8 / spec.packing.num_channels() * std::mem::size_of::<T>() as u8;
            if bits_per_channel > 16 {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                    "Packed bits per channel must be <= 16"));
            }
            match spec.packing {
                ChannelPacking::Gray => image::ColorType::Gray(bits_per_channel),
                ChannelPacking::GrayA => image::ColorType::GrayA(bits_per_channel),
                ChannelPacking::RGBA => image::ColorType::RGBA(bits_per_channel),
            }
        }
    };

    let data = if spec.slicing_dims.plane_dims[0] > spec.slicing_dims.plane_dims[1] {
        // Note, this works correctly because the slab is f-order.
        slab.into_iter().cloned().collect()
    } else {
        slab.into_raw_vec()
    };

    // Get the image data as a byte slice.
    let bytes: &[u8] = unsafe { as_u8_slice(&data) };

    spec.format.encode(writer, bytes, &spec.tile_size, image_color_type)
}

// Get the byte slice of a vec slice in a wrapper function
// so that the lifetime is bound to the original slice's lifetime.
unsafe fn as_u8_slice<T>(s: &[T]) -> &[u8] {
    std::slice::from_raw_parts(
        s as *const [T] as *const u8,
        s.len() * std::mem::size_of::<T>())
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

    /// Cache size (in blocks) per dataset
    #[structopt(long = "ds-block-cache-size", default_value = "1024")]
    ds_block_cache_size: usize,
}

struct AppState {
    n5cache: Arc<N5CacheReader<N5Filesystem>>,
    max_tile_prealloc: usize,
    max_tile_size: TileSize,
}

mod actix_middleware_kludge {
    use actix_service::{IntoTransform, Service, Transform};
    use actix_web::middleware::Condition;

    // Kludge necessary to work around limitations of conditional
    // actix middlewares.
    // See: https://github.com/actix/actix-web/issues/934
    pub(crate) struct WrapCondition<T> {
        trans: T,
        enable: bool,
    }

    impl<T> WrapCondition<T> {
        pub fn new(enable: bool, trans: T) -> Self {
            Self { trans, enable }
        }
    }

    impl<S, T, Target> IntoTransform<Condition<Target>, S> for WrapCondition<T>
    where
        S: Service,
        T: IntoTransform<Target, S>,
        Target: Transform<S, Request = S::Request, Response = S::Response, Error = S::Error>,
    {
        fn into_transform(self) -> Condition<Target> {
            Condition::new(self.enable, self.trans.into_transform())
        }
    }
}

fn main() -> std::io::Result<()> {

    let opt = Options::from_args();
    let root_path = opt.root_path.to_str()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Paths must be UTF-8"))?;
    let n5 = N5Filesystem::open(root_path)?;
    let n5cache = Arc::new(N5CacheReader::wrap(n5, opt.ds_block_cache_size));
    let max_tile_prealloc = opt.max_tile_prealloc;
    let max_tile_size = opt.max_tile_size;
    let cors = opt.cors;
    env_logger::init();

    let mut server = HttpServer::new(
        move || {
            use crate::actix_middleware_kludge::WrapCondition;

            App::new()
                .data(AppState {
                    n5cache: n5cache.clone(),
                    max_tile_prealloc,
                    max_tile_size: max_tile_size.clone(),
                })
                .wrap(actix_web::middleware::Logger::default())
                .service(
                    web::resource("/tile/{spec:.*}")
                        .route(web::get().to(tile))
                )
                .wrap(WrapCondition::new(cors, Cors::new().send_wildcard()))
        })
        .bind((opt.bind_address, opt.port))?;
    if let Some(threads) = opt.threads { server = server.workers(threads); }
    server.run()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_spec_parsing() {
        let ts = TileSpec::from_str("my_test/dataset/0_1/512_512/3/2/1.jpg").unwrap();

        assert_eq!(ts.n5_dataset, "my_test/dataset");
        assert_eq!(ts.slicing_dims, SlicingDims{
            plane_dims: [0u32, 1],
            channel_dim: None,
        });
        assert_eq!(ts.tile_size, TileSize {w: 512, h: 512});
        assert_eq!(ts.coordinates, SmallVec::from([3u64, 2, 1]));
        assert_eq!(ts.format, EncodingFormat::Jpeg(JpegParameters::default()));
    }
}
