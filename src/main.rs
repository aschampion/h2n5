extern crate actix_web;
extern crate image;
extern crate n5;
extern crate ndarray;
extern crate regex;


use std::str::FromStr;

use actix_web::*;


#[derive(Debug, PartialEq)]
struct SlicingDims {
    plane_dims: [u32; 2],
    channel_dim: Option<u32>,
}

#[derive(Debug)]
struct TileSpec {
    n5_dataset: String,
    slicing_dims: SlicingDims,
    tile_size: [u32; 2],
    coordinates: Vec<u64>,
    format: String,
}

impl FromStr for TileSpec {

    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let re = regex::Regex::new(r"^(?P<dataset>.*)/(?P<slicing>\d+_\d+(_\d+)?)/(?P<tile_size>\d+_\d+)(?P<coords>(/\d)+)\.(?P<format>.+)$")
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

        Ok(TileSpec{
            n5_dataset,
            slicing_dims,
            tile_size,
            coordinates,
            format: caps.name("format").unwrap().as_str().into(),
        })
    }
}

#[allow(unknown_lints)]
#[allow(needless_pass_by_value)]
fn tile(req: HttpRequest) -> String {
    format!("{:?}", TileSpec::from_str(&req.match_info()["spec"]))
}

fn main() {
    HttpServer::new(
        || Application::new()
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
        assert_eq!(&ts.format, "jpg");
    }
}
