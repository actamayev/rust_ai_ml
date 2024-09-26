use ndarray::Array2;
use std::error::Error;
use nalgebra::{OMatrix, Dyn};

pub fn invert_matrix(matrix: &Array2<f64>, size: usize) -> Result<Array2<f64>, Box<dyn Error>> {
    // Ensure the matrix is a square
    if matrix.shape() != [size, size] {
        return Err("Matrix must be a square.".into());
    }

    // Convert ndarray::Array2 to nalgebra::MatrixN (9x9 matrix)
    let nalgebra_matrix = OMatrix::<f64, Dyn, Dyn>::from_row_slice(
        size,
        size,
        matrix.as_slice().unwrap()
    );

    // Try to invert the matrix using nalgebra's `try_inverse` method
    let inverted = nalgebra_matrix.try_inverse().ok_or("Matrix is not invertible.")?;

    // Convert nalgebra::MatrixN back to ndarray::Array2
    let inverted_array = Array2::from_shape_vec((size, size), inverted.iter().cloned().collect())?;

    Ok(inverted_array)
}
