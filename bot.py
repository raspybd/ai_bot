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
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
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
import traceback

# تكوين التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_loaders()
        
    def setup_loaders(self):
        self.loaders = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.md': UnstructuredMarkdownLoader
        }

    def process_single_file(self, file_path: Path) -> Optional[List[Any]]:
        try:
            logger.info(f"معالجة الملف: {file_path}")
            
            if not file_path.is_file():
                logger.error(f"الملف غير موجود: {file_path}")
                return None

            file_extension = file_path.suffix.lower()
            if file_extension not in self.config['document_settings']['allowed_extensions']:
                logger.warning(f"نوع الملف غير مدعوم: {file_extension}")
                return None

            loader_class = self.loaders.get(file_extension)
            if not loader_class:
                logger.warning(f"لا يوجد معالج لنوع الملف: {file_extension}")
                return None

            # تحميل المستند
            loader = loader_class(str(file_path))
            docs = loader.load()
            
            # إضافة البيانات الوصفية
            for doc in docs:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_type': file_extension,
                    'creation_date': datetime.fromtimestamp(
                        file_path.stat().st_ctime
                    ).isoformat()
                })

            logger.info(f"تم تحميل {len(docs)} صفحة/قطعة من {file_path}")
            return docs

        except Exception as e:
            logger.error(f"خطأ في معالجة الملف {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def process_documents(self, directory: str) -> List[Any]:
        directory_path = Path(directory)
        if not directory_path.exists():
            logger.error(f"المجلد غير موجود: {directory}")
            return []

        # جمع كل الملفات المدعومة
        files = []
        for ext in self.config['document_settings']['allowed_extensions']:
            files.extend(directory_path.glob(f"**/*{ext}"))

        if not files:
            logger.warning("لم يتم العثور على مستندات في المجلد المحدد")
            return []

        documents = []
        with ThreadPoolExecutor() as executor:
            future_to_file = {
                executor.submit(self.process_single_file, f): f 
                for f in files
            }
            
            with tqdm(total=len(files), desc="تحميل المستندات") as pbar:
                for future in as_completed(future_to_file):
                    try:
                        docs = future.result()
                        if docs:
                            documents.extend(docs)
                    except Exception as e:
                        logger.error(f"خطأ في معالجة المستند: {str(e)}")
                    pbar.update(1)

        return documents

class DocumentBot:
    def __init__(self):
        self.setup_environment()
        self.setup_bot()
        
    def setup_environment(self):
        try:
            # تحميل المتغيرات البيئية
            load_dotenv()
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("لم يتم العثور على مفتاح API")
            
            # تحميل الإعدادات
            with codecs.open('config.yaml', 'r', 'utf-8') as f:
                self.config = yaml.safe_load(f)
            
            logger.info("تم تحميل الإعدادات بنجاح")
        except Exception as e:
            logger.error(f"خطأ في إعداد البيئة: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def setup_bot(self):
        try:
            # تهيئة معالج المستندات
            processor = DocumentProcessor(self.config)
            
            # معالجة المستندات
            documents = processor.process_documents(
                self.config['document_settings']['storage_directory']
            )
            
            if not documents:
                raise ValueError("لم يتم العثور على مستندات صالحة")
            
            logger.info(f"تم تحميل {len(documents)} مستند")

            # تقسيم النصوص
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # قيمة افتراضية إذا لم توجد في الإعدادات
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            
            texts = text_splitter.split_documents(documents)
            logger.info(f"تم إنشاء {len(texts)} قطعة نصية")

            # إعداد قاعدة البيانات المتجهة
            embeddings = OpenAIEmbeddings()
            
            # إنشاء مجلد الحفظ
            persist_dir = "db"
            os.makedirs(persist_dir, exist_ok=True)
            
            # تهيئة قاعدة البيانات
            self.db = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=persist_dir
            )
            
            # إعداد الاسترجاع
            retriever = self.db.as_retriever(
                search_kwargs={
                    "k": self.config['search_settings']['max_results']
                }
            )
            
            # تهيئة نموذج المحادثة
            llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-3.5-turbo",
                max_tokens=1000
            )
            
            # إعداد سلسلة الأسئلة والأجوبة
            self.qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            logger.info("تم إعداد البوت بنجاح")
        except Exception as e:
            logger.error(f"خطأ في إعداد البوت: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def answer_question(self, question: str) -> Dict[str, Any]:
        try:
            if not question:
                return {
                    "answer": "الرجاء إدخال سؤال",
                    "sources": []
                }

            logger.info(f"معالجة السؤال: {question}")
            
            # الحصول على الإجابة
            result = self.qa({"query": question})
            
            # تجهيز المصادر
            sources = []
            seen_contents = set()
            
            for doc in result.get("source_documents", []):
                if doc.page_content in seen_contents:
                    continue
                
                source = {
                    "content": doc.page_content,
                    "metadata": {
                        k: str(v) for k, v in doc.metadata.items()
                        if k in {'source', 'file_type', 'creation_date'}
                    }
                }
                sources.append(source)
                seen_contents.add(doc.page_content)
            
            response = {
                "answer": result.get("result", "لم يتم العثور على إجابة مناسبة"),
                "sources": sources[:5]  # إرجاع أفضل 5 مصادر فقط
            }
            
            logger.info("تم إنشاء الإجابة بنجاح")
            return response
            
        except Exception as e:
            logger.error(f"خطأ في معالجة السؤال: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "answer": f"عذراً، حدث خطأ في معالجة السؤال: {str(e)}",
                "sources": []
            }

def main():
    try:
        print("جاري تحميل البوت...")
        bot = DocumentBot()
        print("تم تحميل البوت بنجاح!")
        
        while True:
            question = input("\nسؤالك (اكتب 'خروج' للإنهاء): ").strip()
            
            if question.lower() in ['خروج', 'exit']:
                print("شكراً لاستخدام البوت!")
                break
                
            if not question:
                print("الرجاء إدخال سؤال")
                continue
            
            response = bot.answer_question(question)
            
            print("\nالإجابة:")
            print(response["answer"])
            
            if response["sources"]:
                print("\nالمصادر:")
                for i, source in enumerate(response["sources"], 1):
                    print(f"\nمصدر {i}:")
                    print(f"المحتوى: {source['content']}")
                    print("المعلومات:")
                    for key, value in source['metadata'].items():
                        print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"خطأ: {str(e)}")
        logger.error(f"خطأ في تشغيل البوت: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
