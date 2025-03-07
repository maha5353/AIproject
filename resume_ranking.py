import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""  # Avoid NoneType errors
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    if not resumes:
        return []  # Return an empty list if no valid resumes are provided

    # Combine job description with resumes
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()  # Assign the transformed vectors

    # Calculate cosine similarity
    job_description_vector = vectors[0]  # Use correct variable name
    resume_vectors = vectors[1:]  # Extract resume vectors
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()

    return cosine_similarities

# Streamlit app
st.title("AI Resume Screening & Candidate Ranking System")

# Custom CSS for Light Green Background
st.markdown(
    """
    <style>
    .stApp {
        background-color: #d4edda;  /* Light Green */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# File uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")

    resumes = []
    valid_files = []  # Keep track of valid files with extracted text

    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        if text.strip():  # Only add non-empty resumes
            resumes.append(text)
            valid_files.append(file.name)

    if resumes:  # Ensure at least one valid resume is processed
        scores = rank_resumes(job_description, resumes)

        # Display scores
        results = pd.DataFrame({"Resume": valid_files, "Score": scores})
        results = results.sort_values(by="Score", ascending=False)

        st.dataframe(results)  # Improved UI rendering
    else:
        st.warning("No valid resumes found. Please upload readable PDFs.")
