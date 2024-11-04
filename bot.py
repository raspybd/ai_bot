import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import yaml
import logging
from datetime import datetime
import codecs
from typing import List, Dict, Any
from pathlib import Path

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Setup log file with proper encoding
log_file = f'logs/bot_{datetime.now().strftime("%Y%m%d")}.log'
with codecs.open(log_file, 'a', 'utf-8'):
    pass

# Setup logging configuration
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

class DocumentLoader:
    """Handle loading of different document types"""
    
    @staticmethod
    def load_documents(directory: str) -> List[Any]:
        """
        Load documents from specified directory supporting multiple file types
        
        Args:
            directory (str): Path to documents directory
            
        Returns:
            List[Any]: List of loaded documents
        """
        documents = []
        
        # Define loaders for different file types
        loaders = {
            ".txt": DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader),
            ".pdf": DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader),
            ".docx": DirectoryLoader(directory, glob="**/*.docx", loader_cls=Docx2txtLoader)
        }
        
        # Load documents using appropriate loader
        for file_type, loader in loaders.items():
            try:
                docs = loader.load()
                documents.extend(docs)
                logging.info(f"Loaded {len(docs)} {file_type} documents")
            except Exception as e:
                logging.error(f"Error loading {file_type} documents: {str(e)}")
        
        return documents

class DocumentBot:
    def __init__(self):
        """Initialize the DocumentBot with environment setup and components"""
        self.setup_environment()
        self.setup_bot()
        
    def setup_environment(self):
        """Setup environment variables and load configuration"""
        try:
            # Load environment variables
            load_dotenv()
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            # Load configuration file
            with codecs.open('config.yaml', 'r', 'utf-8') as f:
                self.config = yaml.safe_load(f)
            
            logging.info("Environment setup completed successfully")
        except Exception as e:
            logging.error(f"Environment setup error: {str(e)}")
            raise

    def setup_bot(self):
        """Setup bot components including document loading and QA chain"""
        try:
            # Load documents using the DocumentLoader
            documents = DocumentLoader.load_documents('./documents')
            if not documents:
                raise ValueError("No documents found in the documents directory")
            
            logging.info(f"Loaded total of {len(documents)} documents")

            # Split texts into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            texts = text_splitter.split_documents(documents)
            logging.info(f"Created {len(texts)} text chunks")

            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings()
            
            # Create persistent storage directory
            persist_directory = "db"
            os.makedirs(persist_directory, exist_ok=True)
            
            # Initialize or load vector store
            self.db = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            self.db.persist()
            
            # Setup retriever with configuration
            retriever = self.db.as_retriever(
                search_kwargs={
                    "k": self.config['search_settings']['max_results'],
                    "fetch_k": self.config['search_settings'].get('fetch_k', 20)
                }
            )
            
            # Initialize QA chain
            self.qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    temperature=0,
                    model_name="gpt-3.5-turbo",
                    streaming=True
                ),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            logging.info("Bot setup completed successfully")
        except Exception as e:
            logging.error(f"Bot setup error: {str(e)}")
            raise

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the QA chain
        
        Args:
            question (str): The question to answer
            
        Returns:
            Dict[str, Any]: Contains the answer and source documents
        """
        try:
            logging.info(f"New question received: {question}")
            result = self.qa({"query": question})
            
            # Prepare sources information
            sources = []
            seen_contents = set()  # To avoid duplicate sources
            
            for doc in result["source_documents"]:
                # Skip if we've already included this content
                if doc.page_content in seen_contents:
                    continue
                    
                source = {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unspecified"),
                    "page": doc.metadata.get("page", "Unspecified")
                }
                sources.append(source)
                seen_contents.add(doc.page_content)
            
            response = {
                "answer": result["result"],
                "sources": sources
            }
            
            logging.info("Answer generated successfully")
            return response
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            logging.error(error_msg)
            return {
                "answer": "Sorry, an error occurred while processing your question.",
                "sources": []
            }

def main():
    """Main function for running the bot in CLI mode"""
    print("Loading bot...")
    try:
        bot = DocumentBot()
        print("\nBot loaded successfully!")
        print("Type 'exit' to quit")
        
        while True:
            question = input("\nYour question: ").strip()
            
            if question.lower() == 'exit':
                print("Thank you for using the bot!")
                break
                
            if not question:
                print("Please enter a valid question")
                continue
            
            response = bot.answer_question(question)
            
            print("\nAnswer:")
            print(response["answer"])
            
            print("\nSources used:")
            for i, source in enumerate(response["sources"], 1):
                print(f"\nSource {i}:")
                print(f"Content: {source['content']}")
                print(f"Source: {source['source']}")
                print(f"Page: {source['page']}")
            
    except Exception as e:
        error_msg = f"Error running bot: {str(e)}"
        print(error_msg)
        logging.error(error_msg)

if __name__ == "__main__":
    main()
