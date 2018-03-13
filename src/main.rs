extern crate actix_web;
extern crate image;
extern crate n5;
extern crate ndarray;
extern crate num_traits;
extern crate regex;
#[macro_use]
extern crate structopt;

use std::io::{
    Write,
};
use std::path::PathBuf;
use std::str::FromStr;

use actix_web::*;
use actix_web::dev::Params;
use actix_web::middleware::cors;
use n5::{
    DatasetAttributes,
    DataType,
    N5Reader,
};
use n5::filesystem::{
    N5Filesystem,
};
use structopt::StructOpt;


#[derive(Debug, PartialEq)]
struct SlicingDims {
    plane_dims: [u32; 2],
    channel_dim: Option<u32>,
}

/// Trait for types that can be configured by URL query string parameters.
trait QueryConfigurable {
    fn configure<'a>(&mut self, params: &'a Params<'a>);
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
    fn configure<'a>(&mut self, params: &'a Params<'a>) {
        if let Ok(q) = params.query::<u8>("q") {
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
        tile_size: &[u32; 2],
        image_color_type: image::ColorType,
    ) -> Result<(), std::io::Error> {
        match *self {
            EncodingFormat::Jpeg(ref p) => {
                let mut encoder = image::jpeg::JPEGEncoder::new_with_quality(writer, p.quality);
                encoder.encode(
                    bytes,
                    tile_size[0],
                    tile_size[1],
                    image_color_type)
            },
            EncodingFormat::Png => {
                let mut encoder = image::png::PNGEncoder::new(writer);
                encoder.encode(
                    bytes,
                    tile_size[0],
                    tile_size[1],
                    image_color_type)
            }
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
    #[allow(single_match)]
    fn configure<'a>(&mut self, params: &'a Params<'a>) {
        match *self {
            EncodingFormat::Jpeg(ref mut p) => p.configure(params),
            _ => (),
        }
    }
}

#[derive(Debug)]
struct TileSpec {
    n5_dataset: String,
    slicing_dims: SlicingDims,
    tile_size: [u32; 2],
    coordinates: Vec<u64>,
    format: EncodingFormat,
}

impl FromStr for TileSpec {

    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let re = regex::Regex::new(r"^(?P<dataset>.*)/(?P<slicing>\d+_\d+(_\d+)?)/(?P<tile_size>\d+_\d+)(?P<coords>(/\d+)+)\.(?P<format>.+)$")
            .expect("TODO: Regex invalid");
        let caps = re.captures(s).expect("TODO: Regex did not match");

        let n5_dataset = caps.name("dataset").unwrap().as_str().into();
        let mut sd_vals = caps.name("slicing").unwrap().as_str().split('_')
            .map(|n| u32::from_str(n).expect("TODO1"));

        let slicing_dims = SlicingDims {
            plane_dims: [sd_vals.next().unwrap(), sd_vals.next().unwrap()],
            channel_dim: sd_vals.next(),
        };

        let mut ts_vals = caps.name("tile_size").unwrap().as_str().split('_')
            .map(|n| u32::from_str(n).expect("TODO2"));

        let tile_size = [ts_vals.next().unwrap(), ts_vals.next().unwrap()];

        let coordinates = caps.name("coords").unwrap().as_str().split('/')
            .filter(|n| !str::is_empty(*n))
            .map(|n| u64::from_str(n).expect("TODO3"))
            .collect();

        Ok(TileSpec {
            n5_dataset,
            slicing_dims,
            tile_size,
            coordinates,
            format: EncodingFormat::from_str(caps.name("format").unwrap().as_str())?,
        })
    }
}

#[allow(unknown_lints)]
#[allow(needless_pass_by_value)]
fn tile(req: HttpRequest<Options>) -> Result<HttpResponse> {
    let spec = {
        let mut spec = TileSpec::from_str(&req.match_info()["spec"]).expect("TODO");
        spec.format.configure(req.query());
        spec
    };

    let n = N5Filesystem::open(req.state().root_path.to_str().expect("TODO: path not unicode?"))?;
    let data_attrs = n.get_dataset_attributes(&spec.n5_dataset)?;
    // Allocate a buffer large enough for the uncompressed tile, as the
    // compressed size will be less with high probability.
    let buffer_size = spec.tile_size[0] as usize * spec.tile_size[1] as usize
        * spec.slicing_dims.channel_dim.map(|d| data_attrs.get_dimensions()[d as usize] as usize).unwrap_or(1)
        * data_attrs.get_data_type().size_of();
    let mut tile_buffer: Vec<u8> = Vec::with_capacity(buffer_size);

    match *data_attrs.get_data_type() {
        DataType::UINT8 => read_and_encode::<u8, _, _>(&n, &data_attrs, &spec, &mut tile_buffer)?,
        DataType::UINT16 => read_and_encode::<u16, _, _>(&n, &data_attrs, &spec, &mut tile_buffer)?,
        _ => return Ok(httpcodes::HttpNotImplemented.with_reason(
                "Data type does not have an image renderer implemented")),
    }
    Ok(HttpResponse::Ok()
        .content_type("image/jpeg")
        .body(tile_buffer)
        .unwrap())
}

// TODO: Single channel only.
fn read_and_encode<T, N: N5Reader, W: Write>(
    n: &N,
    data_attrs: &DatasetAttributes,
    spec: &TileSpec,
    writer: &mut W,
) -> Result<(), std::io::Error>
where n5::VecDataBlock<T>: n5::DataBlock<T>,
      DataType: n5::DataBlockCreator<T>,
      T: Clone + num_traits::identities::Zero {

    // Express the spec tile as an N-dim bounding box.
    let mut size = vec![1i64; data_attrs.get_dimensions().len()];
    size[spec.slicing_dims.plane_dims[0] as usize] = i64::from(spec.tile_size[0]);
    size[spec.slicing_dims.plane_dims[1] as usize] = i64::from(spec.tile_size[1]);
    if let Some(dim) = spec.slicing_dims.channel_dim {
        size[dim as usize] = data_attrs.get_dimensions()[dim as usize];
    }
    let bbox = n5::BoundingBox::new(
        spec.coordinates.iter().map(|n| *n as i64).collect(),
        size,
    );

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
        None => image::ColorType::Gray(8 * std::mem::size_of::<T>() as u8),
    };

    let data = if spec.slicing_dims.plane_dims[0] > spec.slicing_dims.plane_dims[1] {
        // Note, this works correctly because the slab is f-order.
        slab.into_iter().cloned().collect()
    } else {
        slab.into_raw_vec()
    };

    // Get the image data as a byte slice.
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            &data[..] as *const [T] as *const u8,
            data.len() * std::mem::size_of::<T>())
    };

    spec.format.encode(writer, bytes, &spec.tile_size, image_color_type)
        .expect("TODO: encoding");

    Ok(())
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

    /// Number of threads for handling requests.
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
}

fn main() {
    let opt = Options::from_args();

    let mut server = HttpServer::new(
        || {
            let opt = Options::from_args();
            let mut app = Application::with_state(opt.clone())
                .resource("/tile/{spec:.*}", |r| r.f(tile));
            if opt.cors {
                app = app.middleware(cors::Cors::build()
                    .send_wildcard()
                    .finish().expect("Can not create CORS middleware"));
            }
            app
        })
        .bind(format!("{}:{}", opt.bind_address, opt.port)).unwrap();
    if let Some(threads) = opt.threads { server = server.threads(threads); }
    server.run();
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
        assert_eq!(ts.tile_size, [512u32, 512]);
        assert_eq!(ts.coordinates, vec![3u64, 2, 1]);
        assert_eq!(ts.format, EncodingFormat::Jpeg(JpegParameters::default()));
    }
}
