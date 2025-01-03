import streamlit as st
import zipfile
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
import openai


OPENAI_API_KEY = "" 
st.markdown(
    """
    <style>
    .stTextInput input {
        padding-left: 10px;
        padding-top: 10px;
        font-size: 14px;
        height: 40px;  /* Adjust height for the input box */
        border-radius: 5px;  /* Rounded corners */
        border: 2px solid #0072B1;  /* Custom border color */
        width: 100%;  /* Make the input box responsive */
    }

    /* Style the placeholder to act like the label inside the input box */
    .stTextInput input::placeholder {
        color: #999;  /* Placeholder color */
        opacity: 1;  /* Ensure full opacity */
        font-style: italic;  /* Optional: Italicize the placeholder */
        padding-left: 10px;
    }
    </style>
""",unsafe_allow_html=True
)
# Upload ZIP file
st.header("SupplierLens")
st.markdown('<h4><i>Unlocking Insights from Supplier Documents with AI</i></h4>', unsafe_allow_html=True)

with st.sidebar:
    st.title("Your Documents")
    zip_file = st.file_uploader("Upload a ZIP file containing PDFs", type="zip")
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)    
    st.image("lv_logo.png", width=400)

# Extract PDFs from the ZIP file and combine the text
if zip_file:
    
    # Save the uploaded ZIP file temporarily
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        extract_dir = "extracted_files"
        os.makedirs(extract_dir,exist_ok=True)
        zip_ref.extractall(extract_dir)

    # Get all the files in the extracted folder
    extracted_files = os.listdir(extract_dir)
    #st.write(f"Extracted files: {extracted_files}")

    # Collect text from all PDF files
    all_text = ""
    for file_name in extracted_files:
        if file_name.endswith(".PDF"):
            pdf_path = os.path.join(extract_dir, file_name)
            pdf_reader = PdfReader(pdf_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text + "\n\n"  # Combine all text from PDFs

    # If there is no text extracted, inform the user
    if not all_text:
        st.warning("No PDFs found or no text extracted from the PDFs.")
    
    # Break combined text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(all_text)

    # Generate embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Initialize session state for storing responses
    if 'responses' not in st.session_state:
        st.session_state.responses = []

    # Display all previous questions and answers
    if st.session_state.responses:
        for idx, qa in enumerate(st.session_state.responses):
            st.write(f"**Q{idx + 1}:** {qa['question']}")
            st.write(f"**A{idx + 1}:** {qa['answer']}")

    # Get user question after displaying previous answers
    user_question = st.text_input("", key=f"question_{len(st.session_state.responses) + 1}")

    # Perform similarity search and show response
    if user_question:
        # Show spinner while the model processes the request
        with st.spinner('Processing your question...'):
            match = vector_store.similarity_search(user_question)

            # Define the LLM
            llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                temperature=0,
                max_tokens=1000,
                model_name="gpt-3.5-turbo"
            )

            # Chain -> take the question, get relevant documents, pass it to the LLM, generate the output
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=match, question=user_question)

        # Store the new question and answer in the session state
        st.session_state.responses.append({"question": user_question, "answer": response})

    # Display the question-answer pair
    if user_question and response:
        st.write(f"**Q{len(st.session_state.responses)}:** {user_question}")
        st.write(f"**A{len(st.session_state.responses)}:** {response}")

        user_question = st.text_input(" ", key=f"question_{len(st.session_state.responses) + 1}")

