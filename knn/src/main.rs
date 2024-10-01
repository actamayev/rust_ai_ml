use ndarray::Array1;
use std::error::Error;

mod utils;
mod read_knn_csv;
mod error_metrics;

fn cycle_through_housing_data(total_housing_data: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut cloned_total_housing_data = total_housing_data.clone();
    let k = 100;

    for index in 0..cloned_total_housing_data.len() {
        let mut spliced_housing_data = total_housing_data.clone();
        spliced_housing_data.remove(index);
        let mut knn_price_array = Vec::new();

        for record_checking_against in spliced_housing_data {
            if knn_price_array.len() < k {
                if let Some(&price) = record_checking_against.last() {
                    knn_price_array.push(price);
                } else {
                    println!("No price found for the record.");
                }
            } else {
                let distance_to_other_house = find_distance(&cloned_total_housing_data[index], &record_checking_against);
                if let Some(&least_value_in_existing_array) = knn_price_array.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
                    if distance_to_other_house < least_value_in_existing_array {
                        if let Some(position) = knn_price_array.iter().position(|&x| x == least_value_in_existing_array) {
                            knn_price_array.remove(position);
                            knn_price_array.push(distance_to_other_house);
                        }
                    }
                }
            }
            continue
        }

        let average_nearest_house_price = utils::average(&knn_price_array);

        cloned_total_housing_data[index].push(average_nearest_house_price);
    }

    cloned_total_housing_data
}

fn find_distance(focus_house_data: &[f32], comparison_house_data: &[f32]) -> f32 {
    let sq_footage_difference = focus_house_data[0] - comparison_house_data[0];
    let num_bedrooms_difference = focus_house_data[1] - comparison_house_data[1];
    let house_age_difference = focus_house_data[2] - comparison_house_data[2];
    let location_rating_difference = focus_house_data[3] - comparison_house_data[3];

    let sum_of_squares_difference =
        sq_footage_difference.powi(2) +
        num_bedrooms_difference.powi(2) +
        house_age_difference.powi(2) +
        location_rating_difference.powi(2);

    sum_of_squares_difference.sqrt()
}

fn extract_last_two_columns(new_parsed_data: &[Vec<f32>]) -> (Array1<f32>, Array1<f32>) {
    let last_column: Vec<f32> = new_parsed_data.iter()
        .map(|row| row[row.len() - 1])
        .collect();
    
    let second_to_last_column: Vec<f32> = new_parsed_data.iter()
        .map(|row| row[row.len() - 2])
        .collect();
    
    // Convert Vec<f32> to Array1<f32>
    let last_array = Array1::from_vec(last_column);
    let second_to_last_array = Array1::from_vec(second_to_last_column);

    (second_to_last_array, last_array)
}

fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "data/housing_dataset.csv";

    let parsed_housing_data = read_knn_csv::read_knn_csv(file_path)?;

    let new_parsed_data = cycle_through_housing_data(parsed_housing_data);

    let (y_true, y_pred) = extract_last_two_columns(&new_parsed_data);
    let mse = error_metrics::root_mean_squared_error(&y_true, &y_pred);
    let r2 = error_metrics::r2_score(&y_true, &y_pred);

    println!("Mean Squared Error (test): {:?}", mse);
    println!("R² Score (test): {:?}", r2);

    Ok(())
}
