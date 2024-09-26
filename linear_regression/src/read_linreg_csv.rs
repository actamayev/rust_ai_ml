use std::fs::File;
use std::error::Error;
use csv::ReaderBuilder;
use ndarray::{Array2, Array1};

// Function to read the CSV file and extract data as ndarray matrices
pub fn read_linreg_csv(file_path: &str) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
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
