from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()
import os
pinecone_key = os.getenv("PINECONE_API_KEY")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import  RunnablePassthrough, RunnableParallel
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever



class PineconeInsertRetrieval:
    def __init__(self,api_key):
        self.api_key = api_key

    # check index is exist or not 
    def check_index(self,index):
        pc = Pinecone(api_key=self.api_key)
        indexes = pc.list_indexes().names()
        if index not in indexes:
            return "Not Found index"
        elif index in indexes:
            return f"Your index name {index} Found"
    
    # create new index 
    def create_index(self,index_name,dimentions):
        try :
            pc = Pinecone(api_key=self.api_key)
            pc.create_index(
                name=index_name,
                dimension=dimentions,
                metric="cosine",
                spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
                ) 
            )
            print(f"Your index {index_name} created successfull")
            return index_name
        except Exception as ex:
            return f"sorry try again {ex}"
        
    # Delete Index Name
    def delete_index_name(self,index_name):
        try:
            pc = Pinecone(api_key=self.api_key)
            indexes = pc.list_indexes().names()
            if index_name not in indexes:
                return f"Index '{index_name}' does not exist."
            pc.delete_index(index_name)
            return f"Index '{index_name}' deleted successfully."
        except Exception as ex:
            return f"Failed to delete index '{index_name}': {ex}"

    # Delete NameSpace 
    def delete_name_spaces(self,index_name,name_space):
        try:
            # Initialize the index
            pc = Pinecone(api_key=self.api_key)
            index = pc.Index(index_name)
            # Delete the namespace
            response = index.delete(namespace=name_space, delete_all=True)
            if response == {}:
                return f"Namespace '{name_space}' deleted successfully from index '{index_name}'."
            else:
                return f"Unexpected response: {response}"
        except Exception :
            return f"An error occurred: Failed to Delete Namespace"
    
    # Create New nameSpace and insert Data in it 
    def insert_data_in_namespace(self,documents,embeddings,index_name,name_space):
        try:
            doc_search=PineconeVectorStore.from_documents(
                documents,
                embeddings,
                index_name=index_name,
                namespace = name_space
                )
            print(f"Your Name space {name_space} is Created successfully")
            return doc_search
        except Exception as ex:
            return f"Failed to created namespace {ex}"
    
    # Insert Data in Index name 
    def insert_data_in_index(self,documents,embeddings,index_name):
        try:
            PineconeVectorStore.from_documents(
                documents,
                embedding=embeddings,
                index_name=index_name
                )
            print(f"Your Data insert in {index_name} successfully")
        except Exception as ex:
            return f"Failed to created namespace {ex}"

    # Retrieve Data from index name 
    def retrieve_from_index_name(self,index_name,embeddings):
        try:
            vectorstore = PineconeVectorStore.from_existing_index(
                embedding=embeddings,index_name=index_name)
            return vectorstore
        except Exception as ex:
            return f"Failed to load VectorStore {ex}"
        
    # Retrieve Data from Namespace
    def retrieve_from_namespace(self,index_name,embeddings,name_space):
        try:
            vectorstore = PineconeVectorStore.from_existing_index(
                embedding=embeddings,index_name=index_name,namespace=name_space)
            return vectorstore
        except Exception as ex:
            return f"Failed to load VectorStore {ex}"





class SimpleQAChain:
    def __init__(self, model):
        self.model = model

    # Simple QA chain method 
    def QAchain(self, query,template):
        try:
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.model | StrOutputParser()
            input_dict = {"question": query}
            response = chain.invoke(input_dict)
            return response
        except Exception as e:
            return f"Error executing chain: {str(e)}"


    # Simple conversational chain method
    def Conversational_Chain(self, query, history, template):
        try:
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.model | StrOutputParser()
            
            # Create input dict with variables matching the template
            input_dict = {
                "HISTORY": history,
                "QUESTION": query
            }
            
            response = chain.invoke(str(input_dict))
            return response
        except Exception as e:
            return f"Error executing conversational chain: {str(e)}"

    # Qa Retrieval method 
    def QA_Retrieval(self, query, template, vector_store,k):
        try:
            prompt = ChatPromptTemplate.from_template(template)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
            _filter = LLMChainFilter.from_llm(self.model)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=_filter, base_retriever=retriever
            )
            setup_and_retrieval = RunnableParallel(
                {"CONTEXT": compression_retriever, "question": RunnablePassthrough()}
            )
            output_parser = StrOutputParser()
            rag_chain = (
                setup_and_retrieval
                | prompt
                | self.model
                | output_parser
            )
            response = rag_chain.invoke(query)
            return response
        except Exception as e:
            return f"Error executing retrieval chain: {str(e)}"


    # Conversational Retrieval method
    def Conversational_Retrieval(self, query, history, template, vector_store, k):
        try:
            prompt = ChatPromptTemplate.from_template(template)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
            _filter = LLMChainFilter.from_llm(self)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=_filter, base_retriever=retriever
                )
            setup_and_retrieval = RunnableParallel(
                {"CONTEXT": compression_retriever, "QUESTION": RunnablePassthrough(), "HISTORY": RunnablePassthrough()}
            )
            output_parser = StrOutputParser()
            rag_chain = (
                setup_and_retrieval
                | prompt
                | self.model
                | output_parser
            )
            input_dict = {"QUESTION": query, "HISTORY": history}
            response = rag_chain.invoke(str(input_dict))
            return response
        except Exception as e:
            return f"Error executing conversational retrieval chain: {str(e)}"


