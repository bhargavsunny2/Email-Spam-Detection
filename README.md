# 📧 Email Spam Detection System Using Machine Learning

## 📌 About the Project

The **Email Spam Detection System** is a machine learning-based web application that identifies whether a given message is **Spam** or **Not Spam (Normal)**.

The system uses Natural Language Processing (NLP) and the Naive Bayes machine learning algorithm to analyze message text and predict unwanted or suspicious messages.

The application provides a simple, user-friendly interface that can be accessed through laptops, desktops, tablets, and mobile phones using a web browser.

## 🎯 Objectives

- Detect spam and normal messages automatically.
- Reduce unwanted and fraudulent messages.
- Apply machine learning to real-world text classification.
- Provide a simple and responsive web application.
- Display prediction results and model performance statistics.

## ✨ Features

- Spam and normal message detection.
- Machine learning-based prediction.
- Text preprocessing using TF-IDF.
- Prediction confidence display.
- Spam and normal message statistics.
- Model performance evaluation.
- Confusion matrix visualization.
- Spam and normal message charts.
- Prediction history.
- Download prediction history as CSV.
- Clear prediction history.
- Responsive user interface for different screen sizes.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Flask | Web application framework |
| Pandas | Dataset handling |
| Scikit-learn | Machine learning |
| TF-IDF | Text feature extraction |
| Naive Bayes | Spam classification |
| HTML | Webpage structure |
| CSS | Website styling |
| JavaScript | Interactive charts |
| Chart.js | Data visualization |
| GitHub | Source code management |

## 🧠 How It Works

The project follows these steps:

1. **Dataset Collection:** Collect messages labeled as Spam or Normal.
2. **Data Preprocessing:** Clean and prepare message text.
3. **Feature Extraction:** Use TF-IDF to convert text into numerical features.
4. **Model Training:** Train a Multinomial Naive Bayes classifier.
5. **Prediction:** Analyze the user's message and classify it as Spam or Not Spam.
6. **Result Display:** Show the prediction and confidence score on the website.

## 📂 Project Structure

```text
Email-Spam-Detection/
│
├── app.py
├── train_model.py
├── requirements.txt
├── Procfile
│
├── data/
│   └── messages.csv
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Email-Spam-Detection.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### Step 2: Open the Project Folder

```bash
cd Email-Spam-Detection
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Train the Machine Learning Model

```bash
python train_model.py
```

### Step 5: Run the Web Application

```bash
python app.py
```

### Step 6: Open in Your Browser

```text
http://127.0.0.1:5000
```

## 📊 Model Evaluation

The system evaluates its machine learning model using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

These metrics help understand how well the model identifies spam and normal messages.

**Note:** Model performance depends on the size, quality, and balance of the training dataset.

## 📱 Responsive Design

The web application is designed to adjust to different screen sizes, including:

- Desktop computers
- Windows laptops
- MacBooks
- Android smartphones
- iPhones
- Tablets

An internet connection is required when accessing the publicly deployed version.

## 🚀 Future Enhancements

- Integrate a larger real-world email dataset.
- Improve spam detection accuracy.
- Add email file upload functionality.
- Add user authentication.
- Support multiple languages.
- Integrate advanced machine learning algorithms.
- Add real-time email spam detection.

## 🎓 Project Information

**Project Title:** Email Spam Detection System Using Machine Learning

**Domain:** Artificial Intelligence and Machine Learning

**Programming Language:** Python

**Framework:** Flask

**Algorithm:** Multinomial Naive Bayes

**Developer:** K. Bhargav

## 📜 License

This project is developed for educational and academic purposes.
