#[allow(dead_code)]
pub fn min_max_normalize(data: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut normalized_data = data.to_owned(); // Clone the input
    let num_features = data[0].len();
    
    for feature_idx in 0..num_features {
        let min_value = data.iter().map(|row| row[feature_idx]).fold(f32::INFINITY, f32::min);
        let max_value = data.iter().map(|row| row[feature_idx]).fold(f32::NEG_INFINITY, f32::max);

        for row in &mut normalized_data {
            row[feature_idx] = (row[feature_idx] - min_value) / (max_value - min_value);
        }
    }

    normalized_data
}

#[allow(dead_code)]
pub fn z_score_standardize(data: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut standardized_data = data.to_owned();
    
    let num_features = data[0].len();
    
    for feature_idx in 0..num_features {
        let mean: f32 = data.iter().map(|row| row[feature_idx]).sum::<f32>() / data.len() as f32;
        let variance: f32 = data.iter().map(|row| (row[feature_idx] - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std_dev = variance.sqrt();

        for row in &mut standardized_data {
            row[feature_idx] = (row[feature_idx] - mean) / std_dev;
        }
    }

    standardized_data
}
