use std::num::ParseIntError;
use std::str::FromStr;

use n5::ndarray::prelude::*;
use n5::smallvec::smallvec;
use n5::smallvec::SmallVec;
use n5::{
    DatasetAttributes,
    N5Reader,
    ReadableDataBlock,
    ReflectedType,
    ReinitDataBlock,
};

#[derive(Debug, PartialEq)]
pub struct SlicingDims {
    pub plane_dims: [u32; 2],
    pub channel_dim: Option<u32>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TileSize {
    pub w: u32,
    pub h: u32,
}

impl FromStr for TileSize {
    type Err = ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let coords: SmallVec<[&str; 3]> = s.split(',').collect();

        let w = coords[0].parse::<u32>()?;
        let h = coords[1].parse::<u32>()?;

        Ok(TileSize { w, h })
    }
}

impl std::fmt::Display for TileSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{}", self.w, self.h)
    }
}

#[derive(Debug)]
pub struct TileSpec {
    pub n5_dataset: String,
    pub slicing_dims: SlicingDims,
    pub tile_size: TileSize,
    pub coordinates: SmallVec<[u64; 6]>,
}

pub fn read_tile<T, N: N5Reader>(
    n: &N,
    data_attrs: &DatasetAttributes,
    spec: &TileSpec,
) -> Result<ndarray::Array<T, ndarray::Dim<ndarray::IxDynImpl>>, std::io::Error>
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
    n.read_ndarray::<T>(&spec.n5_dataset, data_attrs, &bbox)
}
