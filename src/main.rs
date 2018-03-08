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

#[derive(Debug, PartialEq)]
enum EncodingFormat {
    Jpeg(JpegParameters),
    Png,
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
    let spec = TileSpec::from_str(&req.match_info()["spec"]).expect("TODO");

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
        _ => (),
    }
    Ok(HttpResponse::Ok()
        .content_type("image/jpeg")
        .body(tile_buffer)
        .unwrap())
}

// TODO: u8 only.
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

    let data = slab.into_raw_vec();

    // Get the image data as a byte slice.
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            &data[..] as *const [T] as *const u8,
            data.len() * std::mem::size_of::<T>())
    };

    match spec.format {
        EncodingFormat::Jpeg(ref p) => {
            let mut encoder = image::jpeg::JPEGEncoder::new_with_quality(writer, p.quality);
            encoder.encode(
                bytes,
                spec.tile_size[0],
                spec.tile_size[1],
                image_color_type).expect("TODO: encoding");
        },
        EncodingFormat::Png => {
            let mut encoder = image::png::PNGEncoder::new(writer);
            encoder.encode(
                bytes,
                spec.tile_size[0],
                spec.tile_size[1],
                image_color_type).expect("TODO: encoding");
        }
    }
    Ok(())
}

/// Command line options.
#[derive(StructOpt, Debug, Clone)]
#[structopt(name = "basic")]
struct Options {
    /// N5 root path
    #[structopt(name = "N5_ROOT_PATH", parse(from_os_str), default_value = ".")]
    root_path: PathBuf,
}

fn main() {
    HttpServer::new(
        || Application::with_state(Options::from_args())
            .resource("/tile/{spec:.*}", |r| r.f(tile)))
        .bind("127.0.0.1:8088").unwrap()
        .run();
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
