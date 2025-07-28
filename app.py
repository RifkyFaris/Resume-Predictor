import streamlit as st
import pickle
import docx
import PyPDF2
import re

# Load model, vectorizer, label encoder
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text() or ''
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text

def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

def handle_file_upload(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif ext == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif ext == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type! Upload PDF, DOCX, or TXT.")

def pred(text):
    cleaned = cleanResume(text)
    vec = tfidf.transform([cleaned]).toarray()
    pred_label = svc_model.predict(vec)
    category = le.inverse_transform(pred_label)[0]
    return category

# Glossy, vibrant styling
st.markdown("""
<style>
/* Background Gradient */
body, .reportview-container, .main {
    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
    color: #ffffff;
    font-family: 'Poppins', sans-serif;
}

/* Centered container with glass effect */
.main-container {
    max-width: 700px;
    margin: 3rem auto 4rem;
    background: rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.18);
}

/* Headings */
h1 {
    font-weight: 900;
    font-size: 3.4rem;
    margin-bottom: 0.3rem;
    text-align: center;
    text-shadow: 1px 1px 4px rgba(0,0,0,0.3);
}
h2 {
    font-weight: 700;
    font-size: 1.9rem;
    margin-top: 2.5rem;
    margin-bottom: 1.2rem;
    border-bottom: 3px solid #ffffff80;
    padding-bottom: 0.3rem;
}

/* File uploader */
.stFileUploader > div > div > label > div {
    border: 3px dashed #ffffffaa;
    border-radius: 15px;
    padding: 2.8rem;
    font-weight: 700;
    color: #ffffffcc;
    text-align: center;
    font-size: 1.2rem;
    transition: all 0.3s ease;
    backdrop-filter: blur(8px);
}
.stFileUploader > div > div > label > div:hover {
    background: rgba(255, 255, 255, 0.2);
    border-color: #ffffff;
    cursor: pointer;
}

/* Extracted text area */
textarea {
    background: rgba(255, 255, 255, 0.25) !important;
    border-radius: 15px !important;
    color: #ffffff !important;
    font-size: 1rem !important;
    padding: 1rem !important;
    border: none !important;
    resize: vertical !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    font-family: 'Poppins', sans-serif !important;
}

/* Prediction box */
.prediction-box {
    margin-top: 2rem;
    background: linear-gradient(90deg, #00f260, #0575e6);
    border-radius: 20px;
    padding: 1.6rem 2rem;
    font-size: 1.6rem;
    font-weight: 900;
    text-align: center;
    color: white;
    box-shadow: 0 6px 18px rgba(0, 242, 96, 0.7);
    letter-spacing: 0.06em;
    user-select: none;
}

/* Checkbox label */
div[data-baseweb="checkbox"] > label {
    color: #e0e0e0 !important;
    font-weight: 600;
}

/* Scrollbar styling */
::-webkit-scrollbar {
  width: 8px;
}
::-webkit-scrollbar-track {
  background: transparent;
}
::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.25);
  border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.45);
}
</style>
""", unsafe_allow_html=True)

def main():
    
    st.markdown("<h1> Resume Category Predictor</h1>", unsafe_allow_html=True)
    st.write("Upload your resume (PDF, DOCX, or TXT) below and get an instant job category prediction!")

    uploaded_file = st.file_uploader("", type=["pdf", "docx", "txt"])

    if uploaded_file:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("✅ Text extracted successfully!")

            if st.checkbox("Show extracted text"):
                st.text_area("Extracted Resume Text", resume_text, height=260, key="extracted_text")

            category = pred(resume_text)
            st.markdown(f'<div class="prediction-box">Predicted Category: {category}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
