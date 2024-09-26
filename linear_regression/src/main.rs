use std::error::Error;
use ndarray::{Array2, Array1};

mod error_metrics;
mod read_linreg_csv;
mod train_test_split;
mod lin_alg_operations;

// Function to perform linear regression using the normal equation
fn linear_regression(x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    let xtx = x.t().dot(x);
    let xtx_inv = lin_alg_operations::invert_2x2(&xtx)?;
    let xty = x.t().dot(y);
    let beta = xtx_inv.dot(&xty);
    Ok(beta)
}

fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "data/Hours_of_Study_vs_Exam_Score_Dataset.csv";

    let (x, y) = read_linreg_csv::read_linreg_csv(file_path)?;

    let (x_train, y_train, x_test, y_test) = train_test_split::train_test_split(&x, &y, 0.8);

    // Train the model using the training set
    let beta = linear_regression(&x_train, &y_train)?;

    // Make predictions on the test set
    let y_pred = x_test.dot(&beta);

    // Evaluate the model using error metrics
    let mse = error_metrics::root_mean_squared_error(&y_test, &y_pred);
    let r2 = error_metrics::r2_score(&y_test, &y_pred);

    println!("Mean Squared Error (test): {:?}", mse);
    println!("R² Score (test): {:?}", r2);

    Ok(())
}
