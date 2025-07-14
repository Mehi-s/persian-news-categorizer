# Persian News Categorizer

This project is a text classification model that categorizes Persian news articles into 8 different topics. The model is built using a pipeline that includes a TfidfVectorizer and an XGBoost classifier.

![image](https://github.com/user-attachments/assets/d81a927e-ed6a-4097-a730-843975a09f6d)

## How it Works

The project uses the Persian News dataset, which is a collection of news articles in Persian. The dataset is split into training and testing sets.

### Data Preprocessing

The text data is preprocessed using the `hazm` library, which is a Persian text processing library. The preprocessing steps include:

*   **Stemming:** Reducing words to their root form.
*   **Stopword Removal:** Removing common words that do not add much meaning to the text.

### Model

The model is a pipeline that consists of two main components:

*   **TfidfVectorizer:** This converts the text data into a matrix of TF-IDF features.
*   **XGBoost Classifier:** This is a gradient boosting algorithm that is used to classify the news articles.

## How to Use

To use this project, you need to have Python and the following libraries installed:

*   pandas
*   hazm
*   scikit-learn
*   xgboost

You can install these libraries using pip:

```bash
pip install pandas hazm scikit-learn xgboost
```

Once you have installed the dependencies, you can run the `pip.py` script to train the model and see the classification report:

```bash
python pip.py
```

## Model Performance

The model was trained on 80% of the data and tested on the remaining 20%. The following is the classification report on the test set:

```
              precision    recall  f1-score   support

           0       0.89      0.89      0.89       319
           1       0.91      0.85      0.88       253
           2       0.97      0.94      0.96       363
           3       0.91      0.93      0.92       383
           4       0.96      0.98      0.97       387
           5       0.95      0.96      0.96       385
           6       0.99      0.98      0.98       223
           7       0.91      0.93      0.92       350

    accuracy                           0.94      2663
   macro avg       0.94      0.93      0.93      2663
weighted avg       0.94      0.94      0.94      2663
```

The model achieved an accuracy of 94% on the test set.
