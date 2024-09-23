use ndarray::stack;
use rand::prelude::*;  // Import for shuffling
use ndarray::{Array2, Array1, s, Axis};

pub fn train_test_split(x: &Array2<f64>, y: &Array1<f64>, train_ratio: f64) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    // Generate a list of shuffled indices
    let mut indices: Vec<usize> = (0..x.nrows()).collect();
    indices.shuffle(&mut thread_rng());  // Randomly shuffle the indices

    // Shuffle the data according to the shuffled indices
    let x_shuffled: Array2<f64> = stack(Axis(0), &indices.iter().map(|&i| x.slice(s![i, ..])).collect::<Vec<_>>()).unwrap();
    let y_shuffled: Array1<f64> = Array1::from(indices.iter().map(|&i| y[i]).collect::<Vec<_>>());

    // Compute the training size
    let train_size = (x.nrows() as f64 * train_ratio).round() as usize;

    // Split the shuffled data into training and test sets
    let x_train = x_shuffled.slice(s![..train_size, ..]).to_owned();
    let y_train = y_shuffled.slice(s![..train_size]).to_owned();
    let x_test = x_shuffled.slice(s![train_size.., ..]).to_owned();
    let y_test = y_shuffled.slice(s![train_size..]).to_owned();

    (x_train, y_train, x_test, y_test)
}
