use ndarray::Array1;

pub fn root_mean_squared_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let diff = y_true - y_pred;
    let mse = diff.mapv(|x| x.powi(2)).mean().unwrap();
    mse.sqrt()
}

pub fn r2_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let ss_res = (y_true - y_pred).mapv(|x| x.powi(2)).sum();
    let y_mean = y_true.mean().unwrap();
    let ss_tot = (y_true - y_mean).mapv(|x| x.powi(2)).sum();
    1.0 - (ss_res / ss_tot)
}
