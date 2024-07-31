use linfa::prelude::*;
use linfa_bayes::GaussianNb;
use linfa_preprocessing::CountVectorizer;
use polars::prelude::*;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Load the training dataset from a Parquet file using Polars
    let df = LazyFrame::scan_parquet("data/train-00000-of-00001.parquet", Default::default())?
        .collect()?;

    // Drop unnecessary columns if any
    let df = df.drop("unnecessary_column").unwrap_or(df);

    // Extract the sentence and unfairness level columns
    let sentences: Vec<&str> = df.column("sentence")?
        .utf8()?
        .into_no_null_iter()
        .collect();
    let labels: Vec<i32> = df.column("unfairness_level")?
        .i32()?
        .into_no_null_iter()
        .collect();

    // Split the data into training and test sets
    let split_idx = (sentences.len() as f32 * 0.8) as usize;
    let (train_sentences, test_sentences) = sentences.split_at(split_idx);
    let (train_labels, test_labels) = labels.split_at(split_idx);

    // Vectorize the text data
    let mut vectorizer = CountVectorizer::new();
    let X_train = vectorizer.fit_transform(train_sentences)?;
    let X_test = vectorizer.transform(test_sentences)?;

    // Train the Naive Bayes model
    let model = GaussianNb::params().fit(&Dataset::new(X_train, train_labels.to_vec()))?;

    // Make predictions on the test set
    let y_pred = model.predict(&X_test)?;

    // Evaluate the model accuracy
    let accuracy = y_pred.confusion_matrix(test_labels)?.accuracy();
    println!("Accuracy: {:.4}", accuracy);

    // Predict the unfairness level of a new sentence
    let sample_sentence = "You agree to waive all rights to a jury trial.";
    let sample_vec = vectorizer.transform(&[sample_sentence])?;
    let prediction = model.predict(&sample_vec)?;
    println!("The sentence '{}' is classified as: {}", sample_sentence, prediction[0]);

    Ok(())
}

