import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import yaml
import logging
from datetime import datetime
import codecs
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class DocumentManager:
    """Handle document loading, processing, and metadata management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._setup_loaders()
        
    def _setup_loaders(self):
        """Initialize document loaders for different file types"""
        self.loaders = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".md": UnstructuredMarkdownLoader
        }
        
    def _validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate document metadata against configuration requirements"""
        required_fields = self.config['metadata_fields']['required']
        return all(field in metadata for field in required_fields)
    
    def _validate_category(self, category: str) -> bool:
        """Validate document category against configured categories"""
        return category in self.config['categories']
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for file to check for duplicates"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def load_single_document(self, file_path: str) -> Optional[List[Any]]:
        """Load a single document with appropriate loader"""
        try:
            file_extension = Path(file_path).suffix.lower()
            if file_extension not in self.config['document_settings']['allowed_extensions']:
                logging.warning(f"غير مسموح بنوع الملف: {file_extension}")
                return None
                
            if os.path.getsize(file_path) > self.config['document_settings']['max_file_size']:
                logging.warning(f"حجم الملف كبير جداً: {file_path}")
                return None
                
            loader_class = self.loaders.get(file_extension)
            if not loader_class:
                logging.warning(f"لا يوجد معالج لنوع الملف: {file_extension}")
                return None
                
            loader = loader_class(file_path)
            return loader.load()
            
        except Exception as e:
            logging.error(f"خطأ في تحميل الملف {file_path}: {str(e)}")
            return None

    def load_documents(self, directory: str) -> List[Any]:
        """Load all documents from directory with parallel processing"""
        documents = []
        processed_hashes = set()
        
        # Get all files in directory
        all_files = []
        for ext in self.config['document_settings']['allowed_extensions']:
            all_files.extend(Path(directory).glob(f"**/*{ext}"))
        
        with ThreadPoolExecutor() as executor:
            # Create progress bar
            with tqdm(total=len(all_files), desc="تحميل المستندات") as pbar:
                future_to_file = {
                    executor.submit(self.load_single_document, str(file_path)): file_path 
                    for file_path in all_files
                }
                
                for future in future_to_file:
                    file_path = future_to_file[future]
                    try:
                        docs = future.result()
                        if docs:
                            file_hash = self._get_file_hash(str(file_path))
                            if file_hash not in processed_hashes:
                                documents.extend(docs)
                                processed_hashes.add(file_hash)
                    except Exception as e:
                        logging.error(f"خطأ في معالجة الملف {file_path}: {str(e)}")
                    pbar.update(1)
        
        return documents

class DocumentBot:
    def __init__(self):
        """Initialize DocumentBot with configuration and components"""
        self.setup_environment()
        self.setup_logging()
        self.setup_bot()
        
    def setup_environment(self):
        """Setup environment variables and load configuration"""
        try:
            load_dotenv()
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("لم يتم العثور على مفتاح API")
            
            with codecs.open('config.yaml', 'r', 'utf-8') as f:
                self.config = yaml.safe_load(f)
            
            logging.info("تم تحميل الإعدادات بنجاح")
        except Exception as e:
            logging.error(f"خطأ في إعداد البيئة: {str(e)}")
            raise
            
    def setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        log_dir = log_config.get('directory', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(
            log_dir,
            datetime.now().strftime(log_config.get('file_format', 'bot_%Y%m%d.log'))
        )
        
        logging.basicConfig(
            filename=log_file,
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'),
            encoding=log_config.get('encode', 'utf-8')
        )

    def setup_bot(self):
        """Setup bot components including document loading and QA chain"""
        try:
            # Initialize document manager
            doc_manager = DocumentManager(self.config)
            
            # Load documents
            documents = doc_manager.load_documents(
                self.config['document_settings']['storage_directory']
            )
            
            if not documents:
                raise ValueError("لم يتم العثور على مستندات في المجلد المحدد")
            
            logging.info(f"تم تحميل {len(documents)} مستند")

            # Split texts
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config['document_settings']['chunk_size'],
                chunk_overlap=self.config['document_settings']['chunk_overlap'],
                length_function=len
            )
            texts = text_splitter.split_documents(documents)
            logging.info(f"تم إنشاء {len(texts)} قطعة نصية")

            # Setup vector store
            embeddings = OpenAIEmbeddings()
            vector_store_config = self.config['vector_store']
            
            # Create persistent storage directory
            os.makedirs(vector_store_config['persist_directory'], exist_ok=True)
            
            # Initialize vector store
            self.db = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=vector_store_config['persist_directory'],
                collection_name=vector_store_config['collection_name']
            )
            self.db.persist()
            
            # Setup retriever
            search_settings = self.config['search_settings']
            retriever = self.db.as_retriever(
                search_kwargs={
                    "k": search_settings['max_results'],
                    "fetch_k": search_settings.get('fetch_k', 100),
                    "filter_duplicates": search_settings.get('filter_duplicates', True)
                }
            )
            
            # Setup QA chain
            model_settings = self.config['model_settings']
            self.qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    temperature=model_settings['temperature'],
                    model_name=model_settings['model_name'],
                    max_tokens=model_settings['max_tokens'],
                    streaming=model_settings['streaming']
                ),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            logging.info("تم إعداد البوت بنجاح")
        except Exception as e:
            logging.error(f"خطأ في إعداد البوت: {str(e)}")
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
            logging.info(f"سؤال جديد: {question}")
            result = self.qa({"query": question})
            
            response_format = self.config['response_format']
            max_sources = response_format.get('max_sources', 5)
            
            # Prepare sources information
            sources = []
            seen_contents = set()
            
            for doc in result["source_documents"][:max_sources]:
                if doc.page_content in seen_contents and response_format.get('filter_duplicates', True):
                    continue
                    
                source = {
                    "content": doc.page_content,
                    "metadata": doc.metadata if response_format.get('include_metadata', True) else {}
                }
                sources.append(source)
                seen_contents.add(doc.page_content)
            
            response = {
                "answer": result["result"],
                "sources": sources
            }
            
            logging.info("تم إنشاء الإجابة بنجاح")
            return response
            
        except Exception as e:
            error_msg = f"خطأ في معالجة السؤال: {str(e)}"
            logging.error(error_msg)
            return {
                "answer": "عذراً، حدث خطأ في معالجة سؤالك.",
                "sources": []
            }

def main():
    """Main function for running the bot in CLI mode"""
    print("جاري تحميل البوت...")
    try:
        bot = DocumentBot()
        print("\nتم تحميل البوت بنجاح!")
        print("اكتب 'خروج' للإنهاء")
        
        while True:
            question = input("\nسؤالك: ").strip()
            
            if question.lower() in ['خروج', 'exit']:
                print("شكراً لاستخدام البوت!")
                break
                
            if not question:
                print("الرجاء إدخال سؤال صحيح")
                continue
            
            response = bot.answer_question(question)
            
            print("\nالإجابة:")
            print(response["answer"])
            
            if response["sources"]:
                print("\nالمصادر المستخدمة:")
                for i, source in enumerate(response["sources"], 1):
                    print(f"\nمصدر {i}:")
                    print(f"المحتوى: {source['content']}")
                    if source.get('metadata'):
                        print("البيانات الوصفية:")
                        for key, value in source['metadata'].items():
                            print(f"  {key}: {value}")
            
    except Exception as e:
        error_msg = f"خطأ في تشغيل البوت: {str(e)}"
        print(error_msg)
        logging.error(error_msg)

if __name__ == "__main__":
    main()
