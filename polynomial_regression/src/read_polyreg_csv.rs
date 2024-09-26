use std::fs::File;
use std::error::Error;
use csv::ReaderBuilder;
use ndarray::{Array1, Array2};

// Function to read the CSV file and extract data as ndarray matrices
pub fn read_polyreg_csv(file_path: &str) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
    let file = File::open(file_path)?;

    let mut rdr = ReaderBuilder::new().from_reader(file);

    // Initialize vectors for features (X) and target (z)
    let mut features = Vec::new();
    let mut targets = Vec::new();

    // Read the records from CSV
    for result in rdr.records() {
        let record = result?;

        // Parse each record (assuming hours of study is the first column and exam score the second)
        let x: f64 = record[0].parse()?;
        let y: f64 = record[1].parse()?;
        let z: f64 = record[2].parse()?;

        // Store the features and target values
        features.push(vec![
            1.0,
            x,
            x.powi(2),
            y,
            x * y,
            y * x.powi(2),
            y.powi(2),
            y.powi(2) * x,
            x.powi(2) * y.powi(2)
        ]); // Adding 1.0 for intercept (bias) term
        targets.push(z);
    }

    // Convert vectors to ndarray matrices
    let x_matrix = Array2::from_shape_vec((features.len(), 9), features.concat())?;
    let z = Array1::from_vec(targets);

    Ok((x_matrix, z))
}
