import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
# Import matplotlib.pyplot
import matplotlib.pyplot as plt


# Define the custom layer class
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"), 
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Define the Clean function for text preprocessing
def Clean(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    text = tf.strings.strip(text)
    text = tf.strings.regex_replace(text, '\.\.\.', ' ')
    text = tf.strings.join(['', text, ''], separator=' ')
    return text

# Load the trained model, providing the custom layer definition
model_path = 'model.h5'  # Update this with the actual path to your saved model file
loaded_model = load_model(model_path, custom_objects={'TransformerBlock': TransformerBlock})

# Define the TextVectorization layer
vectorize_layer = TextVectorization(max_tokens=1000, output_mode='int', output_sequence_length=1024)

# Define the Streamlit app
def main():
    st.image('header.png', use_column_width=True)
    st.title("Text Classifier App")
    st.markdown("""
    Welcome to the AI Text Detection App! This tool helps discern between human and AI-generated text. 
    By leveraging machine learning, it analyzes linguistic patterns and semantic features to determine text authenticity. 
    Simply input your text, click 'Classify', and get insights along with a confidence score.
    Ideal for verifying content origins or exploring AI-generated text, this app offers valuable insights into the evolving landscape of textual content.
    """)

    st.subheader("Enter your text below:")
    user_text = st.text_area("Input Text", "")

    if st.button("Classify"):
        preprocessed_text = Clean(user_text)
        vectorize_layer.adapt(tf.constant([user_text]))
        vectorized_text = vectorize_layer(preprocessed_text)
        vectorized_text = np.expand_dims(vectorized_text, axis=0)
        prediction = loaded_model.predict(vectorized_text)
        confidence_score = prediction[0][0]
        binary_prediction = 1 if confidence_score >= 0.5 else 0

        st.write("Prediction:", "Positive" if binary_prediction == 1 else "Negative")

        # Visualize prediction distribution and word distribution side by side
        st.subheader("Prediction and Word Distribution")
        col1, col2 = st.columns(2)

        # Plot prediction distribution
        with col1:
            labels = ['Negative', 'Positive']
            values = [1 - confidence_score, confidence_score]
            fig, ax = plt.subplots()
            ax.bar(labels, values, color=['red', 'green'])
            ax.set_ylabel("Probability")
            ax.set_title("Prediction Distribution")
            st.pyplot(fig)

        # Plot word distribution
        with col2:
            words = user_text.split()
            word_freq = {word: words.count(word) for word in set(words)}
            sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]  # Limit to first 10
            plt.figure()
            plt.bar(*zip(*sorted_word_freq), color='blue')
            plt.xticks(rotation=45)
            plt.xlabel("Words")
            plt.ylabel("Frequency")
            plt.title("Word Distribution (Top 10)")
            st.pyplot(plt)

        # Display confidence score
        st.subheader("Confidence Score")
        st.write("The confidence score for the classification is:")
        st.write(f"**{confidence_score:.2f}**", unsafe_allow_html=True, key='confidence_score')

if __name__ == "__main__":
    main()
