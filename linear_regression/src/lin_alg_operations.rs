use ndarray::Array2;
use std::error::Error;

pub fn invert_2x2(matrix: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
    // Ensure the matrix is 2x2
    if matrix.shape() != [2, 2] {
        return Err("Matrix must be 2x2.".into());
    }

    // Extract elements
    let a = matrix[[0, 0]];
    let b = matrix[[0, 1]];
    let c = matrix[[1, 0]];
    let d = matrix[[1, 1]];

    // Compute the determinant (ad - bc)
    let determinant = a * d - b * c;

    // Ensure the matrix is invertible (determinant != 0)
    if determinant == 0.0 {
        return Err("Matrix is not invertible (determinant is zero).".into());
    }

    // Compute the inverse using the formula
    let inverse = Array2::from_shape_vec(
        (2, 2),
        vec![
            d / determinant,
            -b / determinant,
            -c / determinant,
            a / determinant,
        ],
    )?;

    Ok(inverse)
}
