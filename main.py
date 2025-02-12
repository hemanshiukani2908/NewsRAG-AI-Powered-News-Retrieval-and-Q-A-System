import streamlit as st
import pickle
import time
from PIL import Image
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import torch
from transformers import pipeline
from langchain.chains import RetrievalQA
# Use the loaded pipeline (pipe) with Langchain
from langchain.llms import HuggingFacePipeline

# Load FAISS Index (function)
def load_faiss_index():
    faiss_index_path = "faiss_index"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_index = FAISS.load_local(faiss_index_path, embedding_model,allow_dangerous_deserialization=True)
    return vector_index

# Load LLM pipeline (function)
def load_llm_pipeline():
    # Load the pre-trained model
    model_id = "google/flan-t5-base" # Or any other suitable model ID
    pipe = pipeline(
        "text2text-generation",
        model=model_id,
        device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
        model_kwargs={"torch_dtype": torch.float16}  # Use FP16 for efficiency
    )

    llm = HuggingFacePipeline(pipeline=pipe) 
    return llm  

def main():
    # main app title
    col1, col2 = st.columns([8, 2]) # Create two columns, with the first twice as wide as the second

    with col1:
        st.title("News Research Application")
    with col2:
      image = Image.open("image.png") # Replace with actual path
      st.image(image, width=40)

    # sidebar title
    st.sidebar.title("News Article URLs")

    urls = []
    # url input box
    for i in range(2):
        url=st.sidebar.text_input(f"URL {i+1}")
        urls.append(url)

    # process(extract context from All URL)
    process_url_clicked = st.sidebar.button("Process URLs")

    # Save FAISS index locally
    faiss_index_path = "faiss_index"

    # create empty UI element
    main_placeholder = st.empty()

    if process_url_clicked:
        # 1. Load documents from URLs
        loader = WebBaseLoader(urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()
        
        # 2. divide into chunks because LLM has word limit 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        # As data is of type documents we can directly use split_documents over split_text in order to get the chunks.
        docs = text_splitter.split_documents(data)
        
        # 3. Create Embeddings 
        
        # Load a local embedding model (SBERT)
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create FAISS vector store
        vector_index = FAISS.from_documents(docs, embedding_model)
        
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)
        
        vector_index.save_local(faiss_index_path)
        # Save metadata (optional)
        with open(f"{faiss_index_path}/metadata.pkl", "wb") as f:
            pickle.dump(docs, f)



    query = main_placeholder.text_input("Question: ")

    if query:
        if os.path.exists(faiss_index_path):
                vector_index = load_faiss_index()
                llm = load_llm_pipeline()
                chain = RetrievalQA.from_llm(llm=llm, retriever=vector_index.as_retriever())
                try:
                    docs = vector_index.similarity_search(query)
                    if not docs:
                        print("Not Found!")
                    else:
                        result = chain({"query": query})
                        st.subheader("Answer")
                        st.write(result["result"])
                
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            
if __name__ == "__main__":
    main()
        
        

    