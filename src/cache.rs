use std::collections::HashMap;
use std::io::Error;
use std::sync::RwLock;

use anymap::{any::Any, Map};
use lru_cache::LruCache;
use n5::prelude::*;
use n5::{
    ReadableDataBlock,
    ReinitDataBlock,
};


type DatasetBlockCache<BT> = LruCache<GridCoord, Option<VecDataBlock<BT>>>;

struct BlockCache<BT: ReflectedType> {
    blocks: RwLock<HashMap<String, RwLock<DatasetBlockCache<BT>>>>,
}

impl<BT: ReflectedType> BlockCache<BT> {
    fn new() -> Self {
        BlockCache {
            blocks: RwLock::new(HashMap::new()),
        }
    }

    fn clear(&mut self) {
        self.blocks.write().unwrap().clear();
    }
}

pub struct N5CacheReader<N: N5Reader> {
    reader: N,
    blocks_capacity: usize,
    attr_cache: RwLock<HashMap<String, DatasetAttributes>>,
    cache: Map<dyn Any + Send + Sync>,
}

impl<N: N5Reader> N5CacheReader<N> {
    pub fn wrap(
        reader: N,
        blocks_capacity: usize,
    ) -> Self {
        let mut cache = Map::new();
        cache.insert(BlockCache::<i8>::new());
        cache.insert(BlockCache::<i16>::new());
        cache.insert(BlockCache::<i32>::new());
        cache.insert(BlockCache::<i64>::new());
        cache.insert(BlockCache::<u8>::new());
        cache.insert(BlockCache::<u16>::new());
        cache.insert(BlockCache::<u32>::new());
        cache.insert(BlockCache::<u64>::new());
        cache.insert(BlockCache::<f32>::new());
        cache.insert(BlockCache::<f64>::new());
        Self {
            reader,
            blocks_capacity,
            attr_cache: RwLock::new(HashMap::new()),
            cache,
        }
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.attr_cache.write().unwrap().clear();
        self.cache.get_mut::<BlockCache<i8>>().unwrap().clear();
        self.cache.get_mut::<BlockCache<i16>>().unwrap().clear();
        self.cache.get_mut::<BlockCache<i32>>().unwrap().clear();
        self.cache.get_mut::<BlockCache<i64>>().unwrap().clear();
        self.cache.get_mut::<BlockCache<u8>>().unwrap().clear();
        self.cache.get_mut::<BlockCache<u16>>().unwrap().clear();
        self.cache.get_mut::<BlockCache<u32>>().unwrap().clear();
        self.cache.get_mut::<BlockCache<u64>>().unwrap().clear();
        self.cache.get_mut::<BlockCache<f32>>().unwrap().clear();
        self.cache.get_mut::<BlockCache<f64>>().unwrap().clear();
    }
}

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

    fn exists(&self, path_name: &str) -> Result<bool, Error> {
        self.reader.exists(path_name)
    }

    fn get_block_uri(&self, path_name: &str, grid_position: &[u64]) -> Result<String, Error> {
        self.reader.get_block_uri(path_name, grid_position)
    }

    fn read_block<T>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
    ) -> Result<Option<VecDataBlock<T>>, Error>
            where VecDataBlock<T>: DataBlock<T> + ReadableDataBlock,
                  T: ReflectedType {

        let cache = self.cache.get::<BlockCache<T>>().unwrap();

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

    fn read_block_into<T: ReflectedType, B: DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
        block: &mut B,
    ) -> Result<Option<()>, Error> {

        let cache = self.cache.get::<BlockCache<T>>().unwrap();

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

            if let Some(maybe_block) = ds_block_cache.get_mut(&grid_position[..]) {
                return match maybe_block {
                    Some(existing_block) => {
                        block.reinitialize_with(existing_block);
                        Ok(Some(()))
                    }
                    None => Ok(None)
                };
            }
        }

        let maybe = self.reader.read_block_into(path_name, data_attrs, grid_position.clone(), block)?;
        let ds_cache = &cache.blocks.read().unwrap()[path_name];
        let maybe_block = match maybe {
            Some(_) => {
                Some(VecDataBlock::new(
                    block.get_size().into(),
                    block.get_grid_position().into(),
                    block.get_data().into()))
            }
            None => None
        };
        ds_cache.write().unwrap().insert(grid_position, maybe_block);

        Ok(maybe)
    }

    fn block_metadata(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: &[u64],
    ) -> Result<Option<DataBlockMetadata>, Error> {
        self.reader.block_metadata(path_name, data_attrs, grid_position)
    }

    fn list_attributes(&self, path_name: &str) -> Result<serde_json::Value, Error> {
        self.reader.list_attributes(path_name)
    }
}

impl<N: N5Lister> N5Lister for N5CacheReader<N> {
    fn list(&self, path_name: &str) -> Result<Vec<String>, Error> {
        self.reader.list(path_name)
    }
}
