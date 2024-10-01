use std::fs::File;
use std::error::Error;
use csv::ReaderBuilder;

pub fn read_knn_csv(file_path: &str) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new().from_reader(file);

    let data_to_return: Result<Vec<Vec<f32>>, Box<dyn Error>> = rdr.records()
        .map(|result| {
            let record = result?;
            record.iter()
                .map(|field| field.parse::<f32>().map_err(|e| Box::new(e) as Box<dyn Error>)) // Convert ParseFloatError to Box<dyn Error>
                .collect::<Result<Vec<f32>, _>>() // Collect the parsed fields
        })
        .collect(); // Collect all rows

    data_to_return
}
