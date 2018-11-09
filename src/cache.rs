use std::collections::HashMap;
use std::io::Error;
use std::sync::RwLock;

use lru_cache::LruCache;
use n5::prelude::*;


pub struct N5CacheReader<N: N5Reader> {
    reader: N,
    blocks_capacity: usize,
    cache_u8: BlockCache<u8>,
    cache_f32: BlockCache<f32>,
}

struct BlockCache<BT: Clone> {
    blocks: RwLock<HashMap<String, RwLock<LruCache<Vec<i64>, Option<VecDataBlock<BT>>>>>>,
}

impl<BT: Clone> BlockCache<BT> {
    fn new() -> Self {
        BlockCache {
            blocks: RwLock::new(HashMap::new()),
        }
    }
}

trait TypeCacheable<T: Clone> {
    fn cache(&self) -> &BlockCache<T>;
}

impl<N: N5Reader> N5CacheReader<N> {
    pub fn wrap(
        reader: N,
        blocks_capacity: usize,
    ) -> Self {
        Self {
            reader,
            blocks_capacity,
            cache_u8: BlockCache::new(),
            cache_f32: BlockCache::new(),
        }
    }
}

impl <N: N5Reader, T> TypeCacheable<T> for N5CacheReader<N>
where DataType: n5::DataBlockCreator<T>,
                  VecDataBlock<T>: DataBlock<T>,
                  T: Clone {
    default fn cache(&self) -> &BlockCache<T> {
        unimplemented!()
    }
}

impl<N: N5Reader> TypeCacheable<u8> for N5CacheReader<N> {
    fn cache(&self) -> &BlockCache<u8> {
        &self.cache_u8
    }
}

impl<N: N5Reader> TypeCacheable<f32> for N5CacheReader<N> {
    fn cache(&self) -> &BlockCache<f32> {
        &self.cache_f32
    }
}

impl<N: N5Reader> N5Reader for N5CacheReader<N> {
    fn get_version(&self) -> Result<n5::Version, Error> {
        self.reader.get_version()
    }

    fn get_dataset_attributes(&self, path_name: &str) ->
        Result<n5::DatasetAttributes, Error> {

        self.reader.get_dataset_attributes(path_name)
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
        grid_position: Vec<i64>
    ) -> Result<Option<VecDataBlock<T>>, Error>
            where DataType: n5::DataBlockCreator<T>,
                  VecDataBlock<T>: DataBlock<T>,
                  T: Clone {

        let cache = self.cache();

        if cache.blocks.read().unwrap().get(path_name).is_none() {
            cache.blocks.write().unwrap()
                .entry(path_name.to_owned())
                .or_insert(RwLock::new(LruCache::new(self.blocks_capacity)));
        }

        {
            // Done in a separate scope to not hold any locks while reading
            // blocks on a cache miss.
            // Have to explicitly write these out to satisfy pre-NLL rust borrowck.
            let cache_lock = cache.blocks.read().unwrap();
            let ds_cache = &cache_lock[path_name];
            let mut ds_block_cache = ds_cache.write().unwrap();

            if let Some(block) = ds_block_cache.get_mut(&grid_position) {
                return Ok(block.clone())
            }
        }

        let block = self.reader.read_block(path_name, data_attrs, grid_position.clone())?;
        let ds_cache = &cache.blocks.read().unwrap()[path_name];
        ds_cache.write().unwrap().insert(grid_position, block.clone());

        Ok(block)
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
