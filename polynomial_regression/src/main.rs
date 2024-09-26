use std::error::Error;
use ndarray::{Array2, Array1};

mod error_metrics;
mod train_test_split;
mod read_polyreg_csv;
mod lin_alg_operations;

// Function to perform linear regression using the normal equation
fn polynomial_regression(x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    let xtx = x.t().dot(x);
    let xtx_inv = lin_alg_operations::invert_matrix(&xtx, 9)?;
    let xty = x.t().dot(y);
    let beta = xtx_inv.dot(&xty);
    Ok(beta)
}

fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "data/polynomial_dataset.csv";

    let (x_matrix, z) = read_polyreg_csv::read_polyreg_csv(file_path)?;

    let (x_train, y_train, x_test, y_test) = train_test_split::train_test_split(&x_matrix, &z, 0.8);

    // Train the model using the training set
    let beta = polynomial_regression(&x_train, &y_train)?;

    // Make predictions on the test set
    let y_pred = x_test.dot(&beta);

    // Evaluate the model using error metrics
    let mse = error_metrics::root_mean_squared_error(&y_test, &y_pred);
    let r2 = error_metrics::r2_score(&y_test, &y_pred);

    println!("Mean Squared Error (test): {:?}", mse);
    println!("R² Score (test): {:?}", r2);

    Ok(())
}
