use std::fs::File;
use std::error::Error;
use csv::ReaderBuilder;
use ndarray::{Array2, Array1};

// Function to read the CSV file and extract data as ndarray matrices
fn read_csv(file_path: &str) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
    // Open the file
    let file = File::open(file_path)?;

    // Use the CSV reader
    let mut rdr = ReaderBuilder::new().from_reader(file);

    // Initialize vectors for features (X) and target (y)
    let mut features = Vec::new();
    let mut targets = Vec::new();

    // Read the records from CSV
    for result in rdr.records() {
        let record = result?;

        // Parse each record (assuming hours of study is the first column and exam score the second)
        let hours: f64 = record[0].parse()?;
        let score: f64 = record[1].parse()?;

        // Store the features and target values
        features.push(vec![1.0, hours]); // Adding 1.0 for intercept (bias) term
        targets.push(score);
    }

    // Convert vectors to ndarray matrices
    let x = Array2::from_shape_vec((features.len(), 2), features.concat())?;
    let y = Array1::from_vec(targets);

    Ok((x, y))
}

fn invert_2x2(matrix: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
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

// Function to perform linear regression using the normal equation
fn linear_regression(x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    // X'X
    let xtx = x.t().dot(x);
    println!("xtx, {}", xtx);
    // // (X'X)^-1
    let xtx_inv = invert_2x2(&xtx)?;

    // // X'y
    let xty = x.t().dot(y);

    // // Beta = (X'X)^-1 X'y
    let beta = xtx_inv.dot(&xty);

    Ok(beta)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Path to your CSV file
    let file_path = "data/Hours_of_Study_vs_Exam_Score_Dataset.csv";

    // Step 1: Read the data from CSV
    let (x, y) = read_csv(file_path)?;

    // Step 2: Perform linear regression
    let beta = linear_regression(&x, &y)?;

    // Output the coefficients (intercept and slope)
    println!("Intercept: {}", beta[0]);
    println!("Slope: {}", beta[1]);

    Ok(())
}
