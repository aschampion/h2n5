use std::io::Write;
use std::str::FromStr;

use n5::{
    DatasetAttributes,
    N5Reader,
    ReadableDataBlock,
    ReflectedType,
    ReinitDataBlock,
};

use crate::tiling::read_tile;
use crate::TileImageSpec;

#[derive(Debug, Eq, PartialEq, Hash)]
pub struct JpegParameters {
    pub(crate) quality: u8,
}

impl Default for JpegParameters {
    fn default() -> JpegParameters {
        JpegParameters { quality: 100 }
    }
}

#[derive(Debug, Eq, PartialEq, Hash)]
pub enum EncodingFormat {
    Jpeg(JpegParameters),
    Png,
}

impl EncodingFormat {
    fn encode<W: Write>(
        &self,
        writer: &mut W,
        bytes: &[u8],
        width: u32,
        height: u32,
        image_color_type: image::ColorType,
    ) -> Result<(), image::ImageError> {
        match *self {
            EncodingFormat::Jpeg(ref p) => {
                let mut encoder = image::jpeg::JPEGEncoder::new_with_quality(writer, p.quality);
                encoder.encode(bytes, width, height, image_color_type)
            }
            EncodingFormat::Png => {
                let encoder = image::png::PNGEncoder::new(writer);
                encoder.encode(bytes, width, height, image_color_type)
            }
        }
    }

    pub fn content_type(&self) -> &'static str {
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

#[derive(Debug, Eq, PartialEq, Hash)]
pub enum ChannelPacking {
    Gray,
    GrayA,
    Rgba,
}

impl ChannelPacking {
    fn num_channels(&self) -> u8 {
        match self {
            ChannelPacking::Gray => 1,
            ChannelPacking::GrayA => 2,
            ChannelPacking::Rgba => 4,
        }
    }
}

impl Default for ChannelPacking {
    fn default() -> Self {
        ChannelPacking::Gray
    }
}

impl FromStr for ChannelPacking {
    type Err = (); // TODO

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "gray" => Ok(ChannelPacking::Gray),
            "graya" => Ok(ChannelPacking::GrayA),
            "rgba" => Ok(ChannelPacking::Rgba),
            _ => Err(()),
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum EncodingError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Image(#[from] image::ImageError),
}

pub fn read_and_encode<T, N: N5Reader, W: Write>(
    n: &N,
    data_attrs: &DatasetAttributes,
    spec: &TileImageSpec,
    writer: &mut W,
) -> Result<(), EncodingError>
where
    n5::VecDataBlock<T>: n5::DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock,
    T: ReflectedType + num_traits::identities::Zero,
{
    let tile = read_tile(n, data_attrs, &spec.tile)?;
    encode_tile(tile, spec, writer)
}

// TODO: Single channel only.
pub fn encode_tile<T, W: Write>(
    tile: ndarray::Array<T, ndarray::Dim<ndarray::IxDynImpl>>,
    spec: &TileImageSpec,
    writer: &mut W,
) -> Result<(), EncodingError>
where
    T: ReflectedType,
{
    let image_color_type = match spec.tile.slicing_dims.channel_dim {
        Some(_dim) => {
            // TODO: match RGB/RGBA based on dimensions of dim.
            // Permute slab so that channels dimension is at end.
            unimplemented!()
        }
        None => {
            let bits_per_channel = 8 / spec.packing.num_channels() * std::mem::size_of::<T>() as u8;
            match bits_per_channel {
                // Wow, this is so much better than just specifying my channels and BPC! /s
                8 => match spec.packing {
                    ChannelPacking::Gray => image::ColorType::L8,
                    ChannelPacking::GrayA => image::ColorType::La8,
                    ChannelPacking::Rgba => image::ColorType::Rgba8,
                },
                16 => match spec.packing {
                    ChannelPacking::Gray => image::ColorType::L16,
                    ChannelPacking::GrayA => image::ColorType::La16,
                    ChannelPacking::Rgba => image::ColorType::Rgba16,
                },
                _ => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Packed bits per channel must be 8 or 16",
                    )
                    .into())
                }
            }
        }
    };

    let data = if spec.tile.slicing_dims.plane_dims[0] > spec.tile.slicing_dims.plane_dims[1] {
        // Note, this works correctly because the slab is f-order.
        tile.into_iter().cloned().collect()
    } else {
        tile.into_raw_vec()
    };

    // Get the image data as a byte slice.
    let bytes: &[u8] = unsafe { as_u8_slice(&data) };

    spec.format
        .encode(
            writer,
            bytes,
            spec.tile.tile_size.w,
            spec.tile.tile_size.h,
            image_color_type,
        )
        .map_err(Into::into)
}

// Get the byte slice of a vec slice in a wrapper function
// so that the lifetime is bound to the original slice's lifetime.
unsafe fn as_u8_slice<T>(s: &[T]) -> &[u8] {
    std::slice::from_raw_parts(
        s as *const [T] as *const u8,
        s.len() * std::mem::size_of::<T>(),
    )
}
