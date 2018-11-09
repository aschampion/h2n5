use std::collections::HashMap;
use std::io::Error;
use std::sync::RwLock;

use lru_cache::LruCache;
use n5::prelude::*;


type DatasetBlockCache<BT> = LruCache<GridCoord, Option<VecDataBlock<BT>>>;

struct BlockCache<BT: Clone> {
    blocks: RwLock<HashMap<String, RwLock<DatasetBlockCache<BT>>>>,
}

impl<BT: Clone> BlockCache<BT> {
    fn new() -> Self {
        BlockCache {
            blocks: RwLock::new(HashMap::new()),
        }
    }

    fn clear(&mut self) {
        self.blocks.write().unwrap().clear();
    }
}

trait TypeCacheable<T: Clone> {
    fn cache(&self) -> &BlockCache<T>;
}

pub struct N5CacheReader<N: N5Reader> {
    reader: N,
    blocks_capacity: usize,
    attr_cache: RwLock<HashMap<String, DatasetAttributes>>,
    cache_i8: BlockCache<i8>,
    cache_i16: BlockCache<i16>,
    cache_i32: BlockCache<i32>,
    cache_i64: BlockCache<i64>,
    cache_u8: BlockCache<u8>,
    cache_u16: BlockCache<u16>,
    cache_u32: BlockCache<u32>,
    cache_u64: BlockCache<u64>,
    cache_f32: BlockCache<f32>,
    cache_f64: BlockCache<f64>,
}

impl<N: N5Reader> N5CacheReader<N> {
    pub fn wrap(
        reader: N,
        blocks_capacity: usize,
    ) -> Self {
        Self {
            reader,
            blocks_capacity,
            attr_cache: RwLock::new(HashMap::new()),
            cache_i8: BlockCache::new(),
            cache_i16: BlockCache::new(),
            cache_i32: BlockCache::new(),
            cache_i64: BlockCache::new(),
            cache_u8: BlockCache::new(),
            cache_u16: BlockCache::new(),
            cache_u32: BlockCache::new(),
            cache_u64: BlockCache::new(),
            cache_f32: BlockCache::new(),
            cache_f64: BlockCache::new(),
        }
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.attr_cache.write().unwrap().clear();
        self.cache_i8.clear();
        self.cache_i16.clear();
        self.cache_i32.clear();
        self.cache_i64.clear();
        self.cache_u8.clear();
        self.cache_u16.clear();
        self.cache_u32.clear();
        self.cache_u64.clear();
        self.cache_f32.clear();
        self.cache_f64.clear();
    }
}

// TODO: may be able to remove this specialization-based hack by using
// any `Any`/`TypeId` based approach similar to the `TypeMap` from the
// `polymap` crate.
impl <N: N5Reader, T> TypeCacheable<T> for N5CacheReader<N>
where
                  VecDataBlock<T>: DataBlock<T>,
                  T: ReflectedType {
    default fn cache(&self) -> &BlockCache<T> {
        unimplemented!()
    }
}

macro_rules! type_cacheable {
    ($ty_name:ty, $field:ident) => {
        impl<N: N5Reader> TypeCacheable<$ty_name> for N5CacheReader<N> {
            fn cache(&self) -> &BlockCache<$ty_name> {
                &self.$field
            }
        }
    }
}

type_cacheable!(i8, cache_i8);
type_cacheable!(i16, cache_i16);
type_cacheable!(i32, cache_i32);
type_cacheable!(i64, cache_i64);
type_cacheable!(u8, cache_u8);
type_cacheable!(u16, cache_u16);
type_cacheable!(u32, cache_u32);
type_cacheable!(u64, cache_u64);
type_cacheable!(f32, cache_f32);
type_cacheable!(f64, cache_f64);

impl<N: N5Reader> N5Reader for N5CacheReader<N> {
    fn get_version(&self) -> Result<n5::Version, Error> {
        self.reader.get_version()
    }

    fn get_dataset_attributes(&self, path_name: &str) ->
        Result<n5::DatasetAttributes, Error> {

        {
            if let Some(data_attrs) = self.attr_cache.read().unwrap().get(path_name) {
                return Ok(data_attrs.clone());
            }
        }

        let data_attrs = self.reader.get_dataset_attributes(path_name)?;
        self.attr_cache.write().unwrap().insert(path_name.to_owned(), data_attrs.clone());

        Ok(data_attrs)
    }

    fn exists(&self, path_name: &str) -> bool {
        self.reader.exists(path_name)
    }

    fn get_block_uri(&self, path_name: &str, grid_position: &[i64]) -> Result<String, Error> {
        self.reader.get_block_uri(path_name, grid_position)
    }

    fn read_block<T>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
    ) -> Result<Option<VecDataBlock<T>>, Error>
            where VecDataBlock<T>: DataBlock<T>,
                  T: ReflectedType {

        let cache = self.cache();

        if cache.blocks.read().unwrap().get(path_name).is_none() {
            cache.blocks.write().unwrap()
                .entry(path_name.to_owned())
                .or_insert_with(|| RwLock::new(LruCache::new(self.blocks_capacity)));
        }

        {
            // Done in a separate scope to not hold any locks while reading
            // blocks on a cache miss.
            // Have to explicitly write these out to satisfy pre-NLL rust borrowck.
            let cache_lock = cache.blocks.read().unwrap();
            let ds_cache = &cache_lock[path_name];
            let mut ds_block_cache = ds_cache.write().unwrap();

            if let Some(block) = ds_block_cache.get_mut(&grid_position[..]) {
                return Ok(block.clone())
            }
        }

        let block = self.reader.read_block(path_name, data_attrs, grid_position.clone())?;
        let ds_cache = &cache.blocks.read().unwrap()[path_name];
        ds_cache.write().unwrap().insert(grid_position, block.clone());

        Ok(block)
    }

    fn read_block_into<T: ReflectedType, B: DataBlock<T>>(
        &self,
        _path_name: &str,
        _data_attrs: &DatasetAttributes,
        _grid_position: GridCoord,
        _block: &mut B,
    ) -> Result<Option<()>, Error> {
        unimplemented!()
    }

    fn block_metadata(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: &[i64],
    ) -> Result<Option<DataBlockMetadata>, Error> {
        self.reader.block_metadata(path_name, data_attrs, grid_position)
    }

    fn list(&self, path_name: &str) -> Result<Vec<String>, Error> {
        self.reader.list(path_name)
    }

    fn list_attributes(&self, path_name: &str) -> Result<serde_json::Value, Error> {
        self.reader.list_attributes(path_name)
    }
}
