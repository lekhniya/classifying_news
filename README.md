# 📰 Classifying News Articles

A multi-class news headline classification project that uses **Machine Learning and Deep Learning** techniques to automatically categorize news headlines into seven predefined categories.

## 📌 Project Overview

The project explores different text classification approaches, from traditional machine learning models using TF-IDF features to deep learning models using pretrained GloVe word embeddings. Models are evaluated and compared to identify the most suitable approach for classifying news headlines.

## 🎯 Objectives

- Analyze and preprocess news headline data.
- Explore headline patterns using Exploratory Data Analysis (EDA).
- Convert text into numerical features using TF-IDF and GloVe embeddings.
- Train and compare multiple machine learning and deep learning models.
- Evaluate model performance using classification metrics.
- Deploy the final model through a FastAPI REST API.

## 🗂️ News Categories

The project focuses on seven categories:

- Politics
- Technology
- Entertainment
- Crime
- Queer Voices
- Women
- Travel

## 📊 Dataset

The project uses the **HuffPost News Category Dataset**, which contains news headlines and their associated categories.

The dataset is filtered to retain the seven selected categories. The headline and category columns are used for the classification task.

> **Note:** The dataset files are not included in this repository. Obtain the dataset from its original source and place the JSON file in the project directory before running the notebook.

## ⚙️ Project Workflow

1. **Data Loading:** Load the news dataset from JSON.
2. **Data Analysis:** Examine missing values, duplicate records, category distribution, and headline lengths.
3. **Text Visualization:** Analyze frequent words and generate overall and category-wise word clouds.
4. **Preprocessing:** Clean headlines and encode category labels.
5. **Data Splitting:** Create stratified training, validation, and test sets.
6. **Feature Engineering:** Use TF-IDF with unigrams and bigrams for classical models; use GloVe embeddings and padded sequences for deep learning.
7. **Model Training:** Train and compare the selected models.
8. **Evaluation:** Compare accuracy, precision, recall, weighted F1-score, confusion matrices, and training time.
9. **Model Improvement:** Adjust feature extraction and model architectures, then compare validation performance.
10. **Final Evaluation:** Select the final model using validation performance, retrain it using the training and validation data, and evaluate it on the held-out test set.
11. **Deployment:** Expose headline predictions through a FastAPI application.

## 🤖 Models Implemented

| Model | Text Representation |
|---|---|
| Logistic Regression | TF-IDF |
| Multinomial Naive Bayes | TF-IDF |
| Linear Support Vector Machine (SVM) | TF-IDF |
| Long Short-Term Memory (LSTM) | GloVe embeddings |
| Gated Recurrent Unit (GRU) | GloVe embeddings |

## 📏 Evaluation Metrics

The models are evaluated using:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** How often predicted categories are correct.
- **Recall:** How many actual category instances are correctly identified.
- **Weighted F1-score:** A precision-recall balance that accounts for category support.
- **Confusion Matrix:** Shows correct predictions and category-level misclassifications.
- **Training Time:** Compares the computational cost of model training.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TensorFlow / Keras
- GloVe word embeddings
- Matplotlib and Seaborn
- WordCloud
- FastAPI
- Uvicorn
- Joblib
- Jupyter Notebook

## 📁 Project Structure

```text
Classifying_News_Articles/
├── Classifying_news_arcticles_p1.ipynb
├── newsapp.py
├── News_Category_Dataset_v3.json   # Obtain separately
├── glove.6B.100d.txt              # Obtain separately
└── saved_model/
    ├── final_model.pkl
    ├── tfidf_vectorizer.pkl
    ├── label_encoder.pkl
    └── model_type.pkl
```

The saved model files depend on the final model selected during evaluation. The structure above represents the current TF-IDF and Linear SVM deployment setup.

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/lekhniya/Classifying_News_Articles.git
cd Classifying_News_Articles
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install the required packages

```bash
python -m pip install fastapi "uvicorn[standard]" joblib numpy==1.26.4 pandas==2.2.3 scikit-learn
```

Install Jupyter, TensorFlow, Matplotlib, Seaborn, and WordCloud as needed to run the complete notebook and reproduce the experiments.

### 4. Add the required files

Place the dataset JSON file in the project root. To reproduce the GloVe-based models, obtain the GloVe 100-dimensional embeddings and place `glove.6B.100d.txt` in the project root.

Ensure the saved model artifacts are present in the `saved_model/` directory before starting the API.

## ▶️ Run the FastAPI Application

Start the API from the project root:

```bash
python -m uvicorn newsapp:app --reload
```

Open the interactive API documentation in your browser:

**http://127.0.0.1:8000/docs**

Use the `/predict` endpoint to submit a headline.

Example request:

```json
{
  "headline": "Apple announces a new artificial intelligence product"
}
```

The API returns the submitted headline and its predicted news category.

## 📈 Results

The initial model comparison showed that the **Linear SVM** achieved the strongest baseline validation performance, with approximately **85.68% accuracy** and an **85.30% weighted F1-score**.

After model improvements, Linear SVM was selected as the final model based on the highest weighted F1-score among the compared models. The final test metrics should be reported from the notebook's final evaluation results.

## 🔍 Limitations

- Headlines often provide limited context compared with complete news articles.
- Some categories share vocabulary and may be confused with one another.
- TF-IDF does not capture the full contextual meaning of words.
- GloVe embeddings have limited contextual flexibility compared with modern contextual embeddings.
- The project covers seven selected categories rather than the entire dataset.

## 🔮 Future Improvements

- Experiment with Transformer-based models such as BERT.
- Evaluate contextual embeddings and fine-tuning.
- Improve classification of frequently confused categories.
- Expand the system to support additional news categories.
- Add a user-facing interface and explore production deployment.

## 👩‍💻 Author

**Lekhni**

GitHub: [@lekhniya](https://github.com/lekhniya)

---

*This project demonstrates an end-to-end text classification workflow, from exploratory data analysis and model development to evaluation and API deployment.*
