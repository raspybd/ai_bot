import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import yaml
import logging
from datetime import datetime
import codecs

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
            # Load documents
            loader = DirectoryLoader('./documents', glob="**/*.txt")
            documents = loader.load()
            logging.info(f"Loaded {len(documents)} documents")

            # Split texts into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            texts = text_splitter.split_documents(documents)
            logging.info(f"Created {len(texts)} text chunks")

            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings()
            self.db = Chroma.from_documents(texts, embeddings)
            
            # Setup retriever with configuration
            retriever = self.db.as_retriever(
                search_kwargs={"k": self.config['search_settings']['max_results']}
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

    def answer_question(self, question: str) -> dict:
        """
        Answer a question using the QA chain
        
        Args:
            question (str): The question to answer
            
        Returns:
            dict: Contains the answer and source documents
        """
        try:
            logging.info(f"New question received: {question}")
            result = self.qa({"query": question})
            
            # Prepare sources information
            sources = []
            for doc in result["source_documents"]:
                source = {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unspecified")
                }
                sources.append(source)
            
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
                print(source["content"])
                print(f"Source: {source['source']}")
            
    except Exception as e:
        error_msg = f"Error running bot: {str(e)}"
        print(error_msg)
        logging.error(error_msg)

if __name__ == "__main__":
    main()
