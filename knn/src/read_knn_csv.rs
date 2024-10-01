use std::fs::File;
use std::error::Error;
use csv::ReaderBuilder;

pub fn read_knn_csv(file_path: &str) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let file = File::open(file_path)?;

    let mut rdr = ReaderBuilder::new().from_reader(file);

    let mut data_to_return = Vec::new();

    // Read the records from CSV
    for result in rdr.records() {
        let record = result?;

        let sq_footage = record[0].parse()?;
        let num_bedrooms = record[1].parse()?;
        let house_age = record[2].parse()?;
        let location_rating = record[3].parse()?;
        let house_price = record[4].parse()?;

        // Store the features and target values
        data_to_return.push(vec![
            sq_footage,
            num_bedrooms,
            house_age,
            location_rating,
            house_price
        ]);
    }

    Ok (data_to_return)
}
