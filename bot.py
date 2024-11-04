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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

class DocumentProcessor:
    """Handle document loading and processing with progress tracking"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_loaders()
        
    def setup_loaders(self):
        """Initialize document loaders for different file types"""
        self.loaders = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.md': UnstructuredMarkdownLoader
        }

    def get_file_hash(self, file_path: Path) -> str:
        """Generate MD5 hash of file content"""
        return hashlib.md5(file_path.read_bytes()).hexdigest()

    def process_single_file(self, file_path: Path) -> Optional[List[Any]]:
        """Process a single file and return its documents"""
        try:
            file_size = file_path.stat().st_size
            max_size = self.config['document_settings']['max_file_size']
            file_extension = file_path.suffix.lower()

            # Log file processing
            logging.info(f"معالجة الملف: {file_path.name}")

            # Validate file size
            if file_size > max_size:
                logging.warning(
                    f"حجم الملف كبير: {file_path.name} "
                    f"({file_size / 1048576:.2f}MB > {max_size / 1048576:.2f}MB)"
                )
                return None

            # Get appropriate loader
            loader_class = self.loaders.get(file_extension)
            if not loader_class:
                logging.warning(f"نوع الملف غير مدعوم: {file_extension}")
                return None

            # Load document
            loader = loader_class(str(file_path))
            docs = loader.load()
            
            # Add metadata
            for doc in docs:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_type': file_extension,
                    'file_size': file_size,
                    'creation_date': datetime.fromtimestamp(
                        file_path.stat().st_ctime
                    ).isoformat()
                })

            return docs

        except Exception as e:
            logging.error(f"خطأ في معالجة الملف {file_path}: {str(e)}")
            return None

    def process_documents(self, directory: Path) -> List[Any]:
        """Process all documents in directory with progress tracking"""
        # Get list of files
        files = []
        for ext in self.config['document_settings']['allowed_extensions']:
            files.extend(directory.glob(f"**/*{ext}"))

        if not files:
            logging.warning("لم يتم العثور على مستندات في المجلد المحدد")
            return []

        # Process files with progress bar
        documents = []
        processed_hashes = set()

        with ThreadPoolExecutor() as executor:
            future_to_file = {
                executor.submit(self.process_single_file, f): f 
                for f in files
            }
            
            with tqdm(total=len(files), desc="تحميل المستندات") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        docs = future.result()
                        if docs:
                            file_hash = self.get_file_hash(file_path)
                            if file_hash not in processed_hashes:
                                documents.extend(docs)
                                processed_hashes.add(file_hash)
                    except Exception as e:
                        logging.error(f"فشل في معالجة الملف {file_path}: {str(e)}")
                    pbar.update(1)

        return documents

class DocumentBot:
    """Main bot class for handling document Q&A"""
    
    def __init__(self):
        self.setup_environment()
        self.setup_bot()
        
    def setup_environment(self):
        """Setup environment and load configuration"""
        try:
            # Load environment variables
            load_dotenv()
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("لم يتم العثور على مفتاح API")
            
            # Load configuration
            with codecs.open('config.yaml', 'r', 'utf-8') as f:
                self.config = yaml.safe_load(f)
            
            logging.info("تم تحميل الإعدادات بنجاح")
        except Exception as e:
            logging.error(f"خطأ في إعداد البيئة: {str(e)}")
            raise

    def setup_bot(self):
        """Setup document processor and QA chain"""
        try:
            # Initialize document processor
            processor = DocumentProcessor(self.config)
            
            # Process documents
            documents = processor.process_documents(
                Path(self.config['document_settings']['storage_directory'])
            )
            
            if not documents:
                raise ValueError("لم يتم العثور على مستندات صالحة")
            
            logging.info(f"تم تحميل {len(documents)} مستند")

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config['document_settings']['chunk_size'],
                chunk_overlap=self.config['document_settings']['chunk_overlap'],
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            texts = text_splitter.split_documents(documents)
            logging.info(f"تم إنشاء {len(texts)} قطعة نصية")

            # Setup vector store
            embeddings = OpenAIEmbeddings()
            
            # Create persistent storage
            persist_dir = self.config['vector_store']['persist_directory']
            os.makedirs(persist_dir, exist_ok=True)
            
            # Initialize or load vector store
            self.db = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=persist_dir,
                collection_name=self.config['vector_store']['collection_name']
            )
            self.db.persist()
            
            # Setup retriever
            retriever = self.db.as_retriever(
                search_kwargs={
                    "k": self.config['search_settings']['max_results'],
                    "fetch_k": self.config['search_settings']['fetch_k'],
                    "filter_duplicates": self.config['search_settings']['filter_duplicates']
                }
            )
            
            # Initialize QA chain
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
        """Answer a question using the QA chain"""
        try:
            logging.info(f"سؤال جديد: {question}")
            result = self.qa({"query": question})
            
            # Prepare sources
            sources = []
            seen_contents = set()
            
            for doc in result["source_documents"]:
                # Skip duplicates
                if doc.page_content in seen_contents:
                    continue
                    
                source = {
                    "content": doc.page_content,
                    "metadata": {
                        k: str(v) for k, v in doc.metadata.items()
                        if k in {'source', 'file_type', 'creation_date', 'page'}
                    }
                }
                sources.append(source)
                seen_contents.add(doc.page_content)
            
            response = {
                "answer": result["result"],
                "sources": sources[:self.config['search_settings']['max_results']]
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
    """Main function for CLI operation"""
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
                    print("المعلومات:")
                    for key, value in source['metadata'].items():
                        print(f"  {key}: {value}")
            
    except Exception as e:
        error_msg = f"خطأ في تشغيل البوت: {str(e)}"
        print(error_msg)
        logging.error(error_msg)

if __name__ == "__main__":
    main()
