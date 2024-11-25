import streamlit as st
import tempfile
import os       
from dotenv import load_dotenv
import datetime
from utils import process_jupyter_notebook
load_dotenv()
from langchain_openai import OpenAIEmbeddings
from pinecone_ar_class import PineconeInsertRetrieval, SimpleQAChain
from langchain_openai import ChatOpenAI
import zipfile
from io import BytesIO

# Initialize components
ar_pinecone = PineconeInsertRetrieval(os.getenv("PINECONE_API_KEY"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
index_name = "jupyter-notebook-rag"
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
chainsar = SimpleQAChain(model=llm)

# Configure Streamlit page
st.set_page_config(
    page_title="Jupyter Notebook RAG System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'current_namespace' not in st.session_state:
    st.session_state.current_namespace = None

def process_files(files):
    try:
        all_processed_documents = []
        current_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
        namespace_id = f"jupyter_rag:{current_time}:{index_name}"
        
        for file in files:
            contents = file.read()
            file_extension = file.name.lower().split('.')[-1]

            if not contents:
                st.error(f"Empty file uploaded: {file.name}")
                return None

            if file_extension == 'zip':
                with zipfile.ZipFile(BytesIO(contents)) as zip_ref:
                    for zip_file in zip_ref.namelist():
                        if zip_file.endswith('.ipynb'):
                            with tempfile.NamedTemporaryFile(suffix='.ipynb', delete=False) as tmp_file:
                                tmp_file.write(zip_ref.read(zip_file))
                            
                            try:
                                processed_docs = process_jupyter_notebook(tmp_file.name, orignal_file_name=zip_file)
                                all_processed_documents.extend(processed_docs)
                            finally:
                                try:
                                    os.remove(tmp_file.name)
                                except OSError:
                                    pass
                                
            elif file_extension == 'ipynb':
                with tempfile.NamedTemporaryFile(suffix='.ipynb', delete=False) as tmp_file:
                    tmp_file.write(contents)
                
                try:
                    processed_documents = process_jupyter_notebook(tmp_file.name, orignal_file_name=file.name)
                    all_processed_documents.extend(processed_documents)
                finally:
                    try:
                        os.remove(tmp_file.name)
                    except OSError:
                        pass
            else:
                st.error("Only Jupyter notebook files (.ipynb) or ZIP files containing notebooks are accepted")
                return None

        # Insert documents into Pinecone
        ar_pinecone.insert_data_in_namespace(all_processed_documents, embeddings, index_name, namespace_id)   
        return namespace_id

    except Exception as ex:
        st.error(f"Error processing files: {str(ex)}")
        return None

def query_documents(query, namespace):
    try:
        vectorstore = ar_pinecone.retrieve_from_namespace(
            embeddings=embeddings,
            name_space=namespace,
            index_name=index_name
        )
        prompt_template = """You are assistant. Use the following pieces of {CONTEXT} to generate an answer to the provided question.
        question: {question}.
        Helpful Answer:"""
        results = chainsar.QA_Retrieval(
            template=prompt_template,
            query=query,
            vector_store=vectorstore,
            k=15
        )
        return results
    except Exception as ex:
        st.error(f"Error querying documents: {str(ex)}")
        return None

def delete_namespace(namespace):
    try:
        result = ar_pinecone.delete_name_spaces(index_name=index_name, name_space=namespace)
        return result
    except Exception as ex:
        st.error(f"Error deleting namespace: {str(ex)}")
        return None

# Main UI
st.title("📚 Jupyter Notebook RAG System")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This application allows you to:
    1. Upload Jupyter notebooks (.ipynb) or ZIP files containing notebooks
    2. Query the content using natural language
    3. Manage your document collections
    """)

# Main content
tabs = st.tabs(["Upload", "Query", "Manage"])

# Upload Tab
with tabs[0]:
    st.header("Upload Documents")
    st.markdown("Upload your Jupyter notebooks (.ipynb) or ZIP files containing notebooks.")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['ipynb', 'zip']
    )
    
    if st.button("Process Files", disabled=not uploaded_files):
        with st.spinner("Processing files..."):
            namespace_id = process_files(uploaded_files)
            if namespace_id:
                st.session_state.current_namespace = namespace_id
                st.success(f"Files processed successfully! Namespace: {namespace_id}")

# Query Tab
with tabs[1]:
    st.header("Query Documents")
    
    namespace = st.text_input(
        "Enter Namespace ID",
        value=st.session_state.current_namespace if st.session_state.current_namespace else "",
        placeholder="e.g., jupyter_rag:230415_1430:jupyter-notebook-rag"
    )
    
    query = st.text_area("Enter your question", placeholder="What would you like to know about the notebooks?")
    
    if st.button("Submit Query", disabled=not (namespace and query)):
        with st.spinner("Processing query..."):
            results = query_documents(query, namespace)
            if results:
                st.markdown("### Answer")
                st.write(results)

# Manage Tab
with tabs[2]:
    st.header("Manage Collections")
    
    namespace_to_delete = st.text_input(
        "Enter Namespace ID to Delete",
        placeholder="e.g., jupyter_rag:230415_1430:jupyter-notebook-rag"
    )
    
    if st.button("Delete Namespace", disabled=not namespace_to_delete):
        if st.warning("Are you sure you want to delete this namespace? This action cannot be undone."):
            with st.spinner("Deleting namespace..."):
                result = delete_namespace(namespace_to_delete)
                if result and "successfully" in result:
                    st.success("Namespace deleted successfully!")
                    if st.session_state.current_namespace == namespace_to_delete:
                        st.session_state.current_namespace = None
                else:
                    st.error("Failed to delete namespace")

# Footer
st.markdown("---")
st.markdown("Made by ARZ")