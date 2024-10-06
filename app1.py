#!pip install streamlit openai matplotlib wordcloud pandas
import streamlit as st
import openai
import pandas as pd
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Streamlit App

st.title("Earnings Call Analysis Report with GPT-4")

# Input API key
api_key = st.text_input("Enter OpenAI API Key", type="password")

# File uploader for earnings call transcript
uploaded_file = st.file_uploader("Upload the earnings call transcript file", type=["txt", "csv", "docx"])

# Text area for additional prompts
question = st.text_area("Ask any specific question (optional)")

# Define function for generating the word cloud
def plot_wordcloud(concepts):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(concepts))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Define function for generating the sentiment bar chart
def plot_sentiment_bar(top_10_concepts):
    concepts = [concept['concept'] for concept in top_10_concepts]
    scores = [concept['sentiment_score'] for concept in top_10_concepts]
    
    plt.figure(figsize=(10, 5))
    plt.barh(concepts, scores, color='green')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Concepts')
    plt.title('Top 10 Concepts by Sentiment')
    st.pyplot(plt)

# Button to generate the report
if st.button("Generate Report"):
    if api_key and uploaded_file:
        # Set API key
        openai.api_key = api_key

        # Read uploaded file
        if uploaded_file.name.endswith('.txt'):
            content = uploaded_file.read().decode("utf-8")
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            content = df.to_string()
        else:
            content = uploaded_file.getvalue().decode("utf-8")
        
        # Define the system and user prompts
        system_prompt = (
            "You are an expert financial analyst. "
            "Your task is to analyze the earnings call transcript and provide insights. "
            "Generate results in json template, Do not provide any other details or ```json words etc. Only clean json."
        )
        final_prompt = (
            "Generate a report based on the earnings call and generate the following information in a JSON format. "
            "Sentiment: {'positive': sentiment_score in integer, 'negative': sentiment_score in integer}, "
            "Concepts: provide a list of the top 20 concepts for a word cloud, "
            "Top_10: list the top 10 concepts with their sentiment scores, "
            "Summary: summarize 5 key insights in a list. "
            f"Earnings call transcript:\n{content}"
        )
        
        # OpenAI API call
        try:
            completion = openai.chat.completions.create(
                model="gpt-4o-mini",  # Adjust the model as needed
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_prompt}
                ]
            )

            # Extract the JSON response from the LLM output
            json_response_str = completion.choices[0].message.content
            
            # Convert the extracted text to JSON
            report_data = json.loads(json_response_str)
            
            # Display the report data
            st.subheader("Generated Sentiment Scores")
            st.json(report_data["Sentiment"])

            st.subheader("Top 20 Concepts for Word Cloud")
            plot_wordcloud(report_data["Concepts"])

            st.subheader("Top 10 Concepts by Sentiment")
            plot_sentiment_bar(report_data["Top_10"])

            st.subheader("5 Key Insights Summary")
            for idx, insight in enumerate(report_data["Summary"], start=1):
                st.write(f"{idx}. {insight}")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter your OpenAI API key and upload the transcript file.")
