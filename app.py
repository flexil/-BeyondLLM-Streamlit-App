import streamlit as st
from beyondllm import source, retrieve, embeddings, llms, generator
from beyondllm.embeddings import HuggingFaceEmbeddings
from beyondllm.llms import HuggingFaceHubModel
from beyondllm.retrieve import auto_retriever
from beyondllm.source import fit
import os
import re

hf_token = os.environ['HF_TOKEN']
st.title("BeyondLLM App: Semantic Search, Hybrid Search, and Summarization for AI  Blog Posts Content")
st.write("Perform semantic, hybrid search and summarization to find relevant information within a blog post")
def get_user_input():
    url = st.text_input("Enter the URL of the blog post")
    question = st.text_input("Enter your question")
    search_type = st.selectbox("Select search type", ["Semantic", "Hybrid", "Summarize"])
    return url, question, search_type

def create_retriever(data, embed_model, search_type):
    if search_type == "Semantic":
        return auto_retriever(data=data, embed_model=embed_model, type="cross-rerank", top_k=2)
    elif search_type == "Hybrid":
        return auto_retriever(data=data, embed_model=embed_model, type="hybrid", top_k=5, mode="OR")
    else:
        return None

def generate_output(retriever, llm, prompt):
    pipeline = generator.Generate(question=prompt, retriever=retriever, llm=llm)
    output = pipeline.call()
    return output
    
def summarize(llm, url):
    data = fit(path=url, dtype="url")
    retriever = auto_retriever(data=data, type="summarization", max_length=200)
    summary = retriever.call()
    return summary

def clean_output(output):
    clean_output = re.sub(r'[^a-zA-Z0-9\s]', '', output)
    clean_output = re.sub(r'\s+', ' ', clean_output)
    clean_output = clean_output.replace("RESPONSE ", "")
    clean_output = clean_output.strip()
    return clean_output

def main():
   
    url, question, search_type = get_user_input()
    
    if st.button("Submit"):
        data = fit(path=url, dtype="url")
        embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        retriever = create_retriever(data, embed_model, search_type)
        llm = HuggingFaceHubModel(model="huggingfaceh4/zephyr-7b-alpha", token=hf_token, model_kwargs={"max_new_tokens":256, "temperature": 0.1})
        
        prompt = f"""
        You are a knowledgeable AI assistant.
        You have been trained on a blog post about large language models and machine learning.
        Answer the question based on the content of the blog post if the question is not in the blog say you don't know.
        Question: {question}
        """
        
        if retriever:
            output = generate_output(retriever, llm, prompt)
        else:
            output = summarize(llm, url)
        
        st.write(f"RESPONSE: {clean_output(output)}")

main()
