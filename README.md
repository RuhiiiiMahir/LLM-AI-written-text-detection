# LLM-AI-written-text-detection
This tool helps discern between human and AI-generated text.      By leveraging machine learning, it analyzes linguistic patterns and semantic features to determine text authenticity. This is a Streamlit web application that utilizes a machine learning model to classify text as either human or AI-generated. The app preprocesses the input text, vectorizes it using a TextVectorization layer, and passes it through a trained model to make predictions. Users can input text in the provided text area and click the "Classify" button to get insights along with a confidence score.

![LLM - Ai text detection](LLM-Ai-text-detection.gif)

## Installation
To run this application locally, follow these steps:

### Clone this repository to your local machine:
git clone https://github.com/RuhiiiiMahir/LLM-AI-written-text-detection
### Navigate to the project directory:
cd text-classifier-app
### Install the required dependencies using pip:
pip install -r requirements.txt

## Model Information
The model used in this application was trained as part of a Kaggle competition. You can find the trained model file on Google Drive link https://drive.google.com/drive/folders/1WURlp3hPhR4gXa9w1se1XC_93MZ1bSt_?usp=sharing.

## Usage
After installing the dependencies, you can run the Streamlit app by executing the following command:
streamlit run app.py
This will start the Streamlit server and open the web application in your default web browser.

## Input
The user can input text in the provided text area.

## Output
Upon clicking the "Classify" button, the app will display the following:

Prediction: Whether the input text is classified as "Positive" (human-generated) or "Negative" (AI-generated).
Prediction Distribution: A bar chart showing the probability distribution of the prediction.
Word Distribution: A bar chart displaying the frequency distribution of the top 10 words in the input text.
Confidence Score: The confidence score for the classification.
