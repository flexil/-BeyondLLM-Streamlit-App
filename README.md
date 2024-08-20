# -BeyondLLM-Streamlit-App
BeyondLLM App

A Streamlit app that uses the BeyondLLM library to perform semantic and hybrid search, as well as summarization, on blog posts.

Features

- Input field for URL of blog post
- Input field for question
- Select dropdown for search type (Semantic, Hybrid, Summarize)
- Submit button to generate output
- Output area to display the response

Usage

1. Clone this repository to your local machine
2. Install the required libraries by running pip install -r requirements.txt
3. Run the app using streamlit run streamlit run app.py
4. Open a web browser and navigate to http://localhost:8501
5. Input the URL of a blog post, a question, and select a search type
6. Click the Submit button to generate the output

Requirements

- Python 3.8+
- Streamlit
- BeyondLLM library
- Hugging Face Transformers library

Environment Variables

- HF_TOKEN: your Hugging Face token (required)

License

This app is licensed under the MIT License. 

Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your changes.

Acknowledgments

This app uses the BeyondLLM library, which is built on top of the Hugging Face Transformers library. Thanks to the developers of these libraries for their hard work!
