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
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import yaml
import logging
from datetime import datetime
import codecs
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
import tiktoken

# تكوين التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# إعداد القالب بالعربية
CUSTOM_PROMPT_TEMPLATE = """استخدم المعلومات التالية للإجابة على السؤال. إذا لم تكن تعرف الإجابة، قل فقط أنك لا تعرف، لا تحاول اختلاق إجابة.

{context}

السؤال: {question}

الإجابة بالعربية:"""

class DocumentBot:
    def __init__(self):
        self.setup_environment()
        self.setup_bot()
        
    def setup_environment(self):
        try:
            load_dotenv()
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("لم يتم العثور على مفتاح API")
            
            with codecs.open('config.yaml', 'r', 'utf-8') as f:
                self.config = yaml.safe_load(f)
            
            logger.info("تم تحميل الإعدادات بنجاح")
        except Exception as e:
            logger.error(f"خطأ في إعداد البيئة: {str(e)}")
            raise

    def setup_bot(self):
        try:
            # تحميل المستندات
            documents = self.load_documents()
            if not documents:
                raise ValueError("لم يتم العثور على مستندات صالحة")
            
            logger.info(f"تم تحميل {len(documents)} مستند")

            # تقسيم النصوص مع حجم أصغر
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # تقليل حجم القطع
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            
            texts = text_splitter.split_documents(documents)
            logger.info(f"تم إنشاء {len(texts)} قطعة نصية")

            # إعداد قاعدة البيانات المتجهة
            embeddings = OpenAIEmbeddings()
            persist_dir = "db"
            os.makedirs(persist_dir, exist_ok=True)
            
            self.db = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=persist_dir
            )
            
            # إعداد استرجاع مع عدد أقل من النتائج
            retriever = self.db.as_retriever(
                search_kwargs={
                    "k": 3  # تقليل عدد النتائج
                }
            )
            
            # تهيئة نموذج المحادثة مع إعدادات محسنة
            llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-3.5-turbo-16k",  # استخدام نموذج يدعم نصوص أطول
                max_tokens=1000,
                request_timeout=30
            )

            # إعداد القالب المخصص
            prompt = PromptTemplate(
                template=CUSTOM_PROMPT_TEMPLATE,
                input_variables=["context", "question"]
            )
            
            # إعداد سلسلة الأسئلة والأجوبة
            self.qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={
                    "prompt": prompt,
                    "verbose": True
                },
                return_source_documents=True
            )
            
            logger.info("تم إعداد البوت بنجاح")
        except Exception as e:
            logger.error(f"خطأ في إعداد البوت: {str(e)}")
            raise

    def load_documents(self):
        directory = self.config['document_settings']['storage_directory']
        allowed_extensions = self.config['document_settings']['allowed_extensions']
        
        documents = []
        for ext in allowed_extensions:
            try:
                loader = DirectoryLoader(
                    directory,
                    glob=f"**/*{ext}",
                    loader_cls=self.get_loader_class(ext)
                )
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"تم تحميل {len(docs)} مستند من النوع {ext}")
            except Exception as e:
                logger.error(f"خطأ في تحميل المستندات من النوع {ext}: {str(e)}")
        
        return documents

    def get_loader_class(self, extension):
        loaders = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.md': UnstructuredMarkdownLoader
        }
        return loaders.get(extension, TextLoader)

    def answer_question(self, question: str) -> Dict[str, Any]:
        try:
            if not question:
                return {
                    "answer": "الرجاء إدخال سؤال",
                    "sources": []
                }

            logger.info(f"معالجة السؤال: {question}")
            
            result = self.qa({"query": question})
            
            sources = []
            seen_contents = set()
            
            for doc in result.get("source_documents", [])[:3]:  # تحديد عدد المصادر
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
                "sources": sources
            }
            
            logger.info("تم إنشاء الإجابة بنجاح")
            return response
            
        except Exception as e:
            logger.error(f"خطأ في معالجة السؤال: {str(e)}")
            return {
                "answer": "عذراً، حدث خطأ في معالجة السؤال. الرجاء المحاولة مرة أخرى بسؤال أقصر أو أكثر تحديداً.",
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

if __name__ == "__main__":
    main()
