
use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::BufReader;

use jpeg_decoder::{Decoder,PixelFormat};
use ndarray_vision::core::{RGB, Image, Gray};

fn main() {
    let root = Path::new("C:/Users/admin/Desktop/codespace/python/Structure-from-Motion/dataset/box");
    let test_image = root.clone().join("IMG_20200328_172713.jpg");
    println!("{:?}", test_image);
        
        
    let file = File::open(test_image).expect("failed to open file");
    let mut decoder = Decoder::new(BufReader::new(file));
    let pixels = decoder.decode().expect("failed to decode image");
    let metadata = decoder.info().unwrap();    

    println!("pixel_format from Metatdata {:?}",metadata.pixel_format);
    println!("width from Metatdata {:?}",metadata.width);
    println!("height from Metatdata {:?}",metadata.height);

    assert_eq!(
        metadata.pixel_format,
        PixelFormat::RGB24,
        "Colour format is wrong"
    );
    let image = Image::<u8, RGB>::from_shape_data(metadata.height as usize, metadata.width as usize, pixels);
    assert_eq!(
        metadata.height,
        image.rows() as u16,
        "Height  is wrong"
    );

    assert_eq!(
        metadata.width,
        image.cols() as u16,
        "Width  is wrong"
    );

    println!("Image size {}X{}",image.cols(),image.rows());

    let gray_image=Image::<u8, Gray>::from(image);

    println!("Gray Images chennal {}",gray_image.channels());

    //let gray_image_path = PathBuf::from("./gray_image.jpg");
}