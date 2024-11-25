from typing import Dict, List, Tuple
from langchain.schema import Document
from langchain_community.document_loaders import NotebookLoader
import re

def process_jupyter_notebook(file_path: str, max_chunk_length: int = 1000,orignal_file_name:str=None) -> List[Document]:
    """
    Process a Jupyter notebook file and return a list of Document objects.
    
    Args:
        file_path (str): Path to the Jupyter notebook file
        max_chunk_length (int): Maximum length for content chunks (default: 1000)
    
    Returns:
        List[Document]: List of processed documents with metadata
    """
    
    def load_notebook(file_path: str) -> List[Document]:
        """Load notebook using NotebookLoader"""
        try:
            loader = NotebookLoader(
                file_path,
                include_outputs=True,
                max_output_length=100,
                remove_newline=True,
            )
            return loader.load()
        except Exception as e:
            raise ValueError(f"Error loading notebook: {str(e)}")

    def split_cells(data: List[Document]) -> List[Tuple[str, str]]:
        """Split notebook into cells with their types"""
        cells = []
        for cell in data:
            cell_type = re.search(r"'(.*?)'", cell.page_content).group(1)
            content = re.sub(r"\\'", "'", cell.page_content)
            cells.append((cell_type, content))
        return cells

    def chunk_content(content: str, max_length: int) -> List[str]:
        """Split content into smaller chunks if needed"""
        chunks = []
        current_chunk = ""
        
        lines = content.split('\n')
        
        for line in lines:
            if len(current_chunk) + len(line) > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line
            else:
                current_chunk = current_chunk + '\n' + line if current_chunk else line
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks if chunks else [content]

    def create_documents(cells: List[Tuple[str, str]], orignal_file_name: str, max_length: int) -> List[Document]:
        """Create Document objects from cells"""
        documents = []
        for cell_type, content in cells:
            if len(content) > max_length:
                # Split into smaller chunks if content is too long
                cell_chunks = chunk_content(content, max_length)
                for i, chunk in enumerate(cell_chunks):
                    if chunk.strip():  # Ignore empty chunks
                        document = Document(
                            page_content=chunk,
                            metadata={
                                "source": orignal_file_name,
                                "cell_type": cell_type,
                                "chunk_index": i,
                            }
                        )
                        documents.append(document)
            else:
                # Keep cell as is if content is short enough
                if content.strip():
                    document = Document(
                        page_content=content,
                        metadata={
                            "source": orignal_file_name,
                            "cell_type": cell_type,
                            "chunk_index": 0,
                        }
                    )
                    documents.append(document)
        return documents

    try:
        # Process the notebook in steps
        notebook_data = load_notebook(file_path)
        cells = split_cells(notebook_data)
        documents = create_documents(cells,orignal_file_name, max_chunk_length)
        return documents
    
    except Exception as e:
        raise Exception(f"Error processing notebook: {str(e)}")