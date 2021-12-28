use ndarray::{Array3, Ix3};
use ndarray_vision::core::*;
use ndarray_vision::format::netpbm::*;
use ndarray_vision::format::*;
use ndarray_vision::processing::*;
use std::path::{Path, PathBuf};

fn main() {
    let root = Path::new("C:/Users/admin/Desktop/codespace/python/Structure-from-Motion/dataset/box");
    let cameraman = root.clone().join("IMG_20200328_172713.jpg");
    println!("{:?}", cameraman);

    let decoder = PpmDecoder::default();
    let image: Image<u8, _> = decoder
        .decode_file(cameraman)
        .expect("Couldn't open cameraman.ppm");

    let boxkern: Array3<f64> =
        BoxLinearFilter::build(Ix3(3, 3, 3)).expect("Was unable to construct filter");

    let mut image: Image<f64, _> = image.into_type();

    let _ = image
        .conv2d_inplace(boxkern.view())
        .expect("Poorly sized kernel");
    // There's no u8: From<f64> so I've done this to hack things

    let mut cameraman = PathBuf::from(&root);
    cameraman.push("images/cameramanblur.ppm");

    let ppm = PpmEncoder::new_plaintext_encoder();
    ppm.encode_file(&image, cameraman)
        .expect("Unable to encode ppm");
}