use std::io::Write;
use std::str::FromStr;

use n5::ndarray::prelude::*;
use n5::smallvec::smallvec;
use n5::{
    DatasetAttributes,
    N5Reader,
    ReadableDataBlock,
    ReflectedType,
    ReinitDataBlock,
};

use crate::{
    TileSize,
    TileSpec,
};

#[derive(Debug, PartialEq)]
pub struct JpegParameters {
    pub(crate) quality: u8,
}

impl Default for JpegParameters {
    fn default() -> JpegParameters {
        JpegParameters { quality: 100 }
    }
}

#[derive(Debug, PartialEq)]
pub enum EncodingFormat {
    Jpeg(JpegParameters),
    Png,
}

impl EncodingFormat {
    fn encode<W: Write>(
        &self,
        writer: &mut W,
        bytes: &[u8],
        tile_size: TileSize,
        image_color_type: image::ColorType,
    ) -> Result<(), image::ImageError> {
        match *self {
            EncodingFormat::Jpeg(ref p) => {
                let mut encoder = image::jpeg::JPEGEncoder::new_with_quality(writer, p.quality);
                encoder.encode(bytes, tile_size.w, tile_size.h, image_color_type)
            }
            EncodingFormat::Png => {
                let encoder = image::png::PNGEncoder::new(writer);
                encoder.encode(bytes, tile_size.w, tile_size.h, image_color_type)
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

#[derive(Debug, PartialEq)]
pub enum ChannelPacking {
    Gray,
    GrayA,
    RGBA,
}

impl ChannelPacking {
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
    type Err = (); // TODO

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "gray" => Ok(ChannelPacking::Gray),
            "graya" => Ok(ChannelPacking::GrayA),
            "rgba" => Ok(ChannelPacking::RGBA),
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

// TODO: Single channel only.
pub fn read_and_encode<T, N: N5Reader, W: Write>(
    n: &N,
    data_attrs: &DatasetAttributes,
    spec: &TileSpec,
    writer: &mut W,
) -> Result<(), EncodingError>
where
    n5::VecDataBlock<T>: n5::DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock,
    T: ReflectedType + num_traits::identities::Zero,
{
    // Express the spec tile as an N-dim bounding box.
    let mut size = smallvec![1u64; data_attrs.get_dimensions().len()];
    size[spec.slicing_dims.plane_dims[0] as usize] = u64::from(spec.tile_size.w);
    size[spec.slicing_dims.plane_dims[1] as usize] = u64::from(spec.tile_size.h);
    if let Some(dim) = spec.slicing_dims.channel_dim {
        size[dim as usize] = data_attrs.get_dimensions()[dim as usize];
    }
    let bbox = BoundingBox::new(spec.coordinates.clone(), size);

    // Read the N-dim slab of blocks containing the tile from N5.
    let slab = n.read_ndarray::<T>(&spec.n5_dataset, data_attrs, &bbox)?;

    let image_color_type = match spec.slicing_dims.channel_dim {
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
                    ChannelPacking::RGBA => image::ColorType::Rgba8,
                },
                16 => match spec.packing {
                    ChannelPacking::Gray => image::ColorType::L16,
                    ChannelPacking::GrayA => image::ColorType::La16,
                    ChannelPacking::RGBA => image::ColorType::Rgba16,
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

    let data = if spec.slicing_dims.plane_dims[0] > spec.slicing_dims.plane_dims[1] {
        // Note, this works correctly because the slab is f-order.
        slab.into_iter().cloned().collect()
    } else {
        slab.into_raw_vec()
    };

    // Get the image data as a byte slice.
    let bytes: &[u8] = unsafe { as_u8_slice(&data) };

    spec.format
        .encode(writer, bytes, spec.tile_size, image_color_type)
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
