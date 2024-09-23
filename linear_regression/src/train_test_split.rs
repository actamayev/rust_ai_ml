use ndarray::{Array2, Array1, s};

pub fn train_test_split(x: &Array2<f64>, y: &Array1<f64>, train_ratio: f64) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let train_size = (x.nrows() as f64 * train_ratio).round() as usize;
    
    let x_train = x.slice(s![..train_size, ..]).to_owned();
    let y_train = y.slice(s![..train_size]).to_owned();
    let x_test = x.slice(s![train_size.., ..]).to_owned();
    let y_test = y.slice(s![train_size..]).to_owned();

    (x_train, y_train, x_test, y_test)
}
