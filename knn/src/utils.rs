pub fn average(values: &[f32]) -> f32 {
    let sum: f32 = values.iter().sum();
    let count = values.len();

    if count == 0 {
        return 0.0;  // Handle empty vector case to avoid division by zero
    }

    sum / count as f32
}
