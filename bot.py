from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import yaml
import logging
from datetime import datetime
import codecs  # إضافة مكتبة للتعامل مع الترميز

# إنشاء مجلد السجلات إذا لم يكن موجوداً
os.makedirs('logs', exist_ok=True)

# إعداد ملف السجلات مع الترميز المناسب
log_file = f'logs/bot_{datetime.now().strftime("%Y%m%d")}.log'
# إنشاء ملف السجلات بترميز UTF-8
with codecs.open(log_file, 'a', 'utf-8'):
  pass

# إعداد التسجيل
logging.basicConfig(
  filename=log_file,
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s'
)

class DocumentBot:
  def __init__(self):
      self.setup_environment()
      self.setup_bot()
      
  def setup_environment(self):
      """إعداد البيئة وتحميل الإعدادات"""
      try:
          # تحميل متغيرات البيئة
          load_dotenv()
          self.api_key = os.getenv('OPENAI_API_KEY')
          if not self.api_key:
              raise ValueError("لم يتم العثور على مفتاح API")
          
          # تحميل ملف الإعدادات
          with codecs.open('config.yaml', 'r', 'utf-8') as f:
              self.config = yaml.safe_load(f)
          
          logging.info("تم تحميل الإعدادات بنجاح")
      except Exception as e:
          logging.error(f"خطأ في إعداد البيئة: {str(e)}")
          raise

  def setup_bot(self):
      """إعداد مكونات البوت"""
      try:
          # تحميل المستندات
          loader = DirectoryLoader('./documents', glob="**/*.txt")
          documents = loader.load()
          logging.info(f"تم تحميل {len(documents)} مستند")

          # تقسيم النصوص
          text_splitter = RecursiveCharacterTextSplitter(
              chunk_size=1000,
              chunk_overlap=200,
              length_function=len
          )
          texts = text_splitter.split_documents(documents)

          # إنشاء embeddings
          embeddings = OpenAIEmbeddings()
          
          # إنشاء قاعدة البيانات المتجهة
          self.db = Chroma.from_documents(texts, embeddings)
          
          # إعداد نموذج المحادثة
          retriever = self.db.as_retriever(
              search_kwargs={"k": self.config['search_settings']['max_results']}
          )
          
          self.qa = RetrievalQA.from_chain_type(
              llm=ChatOpenAI(
                  temperature=0,
                  model_name="gpt-3.5-turbo"
              ),
              chain_type="stuff",
              retriever=retriever,
              return_source_documents=True
          )
          
          logging.info("تم إعداد البوت بنجاح")
      except Exception as e:
          logging.error(f"خطأ في إعداد البوت: {str(e)}")
          raise

  def answer_question(self, question: str) -> dict:
      """الإجابة على سؤال"""
      try:
          logging.info(f"سؤال جديد: {question}")
          result = self.qa({"query": question})
          
          # تجهيز المصادر
          sources = []
          for doc in result["source_documents"]:
              source = {
                  "content": doc.page_content,
                  "source": doc.metadata.get("source", "غير محدد")
              }
              sources.append(source)
          
          response = {
              "answer": result["result"],
              "sources": sources
          }
          
          logging.info("تم إرسال الإجابة بنجاح")
          return response
          
      except Exception as e:
          logging.error(f"خطأ في معالجة السؤال: {str(e)}")
          return {
              "answer": "عذراً، حدث خطأ في معالجة سؤالك",
              "sources": []
          }

def main():
  """الدالة الرئيسية لتشغيل البوت"""
  print("جاري تحميل البوت...")
  try:
      bot = DocumentBot()
      print("\nتم تحميل البوت بنجاح!")
      print("اكتب 'خروج' للإنهاء")
      
      while True:
          question = input("\nسؤالك: ").strip()
          
          if question.lower() == 'خروج':
              print("شكراً لاستخدام البوت!")
              break
              
          if not question:
              print("الرجاء إدخال سؤال صحيح")
              continue
          
          response = bot.answer_question(question)
          
          print("\nالإجابة:")
          print(response["answer"])
          
          print("\nالمصادر المستخدمة:")
          for i, source in enumerate(response["sources"], 1):
              print(f"\nمصدر {i}:")
              print(source["content"])
              print(f"المصدر: {source['source']}")
          
  except Exception as e:
      print(f"حدث خطأ: {str(e)}")
      logging.error(f"خطأ في تشغيل البوت: {str(e)}")

if __name__ == "__main__":
  main()
