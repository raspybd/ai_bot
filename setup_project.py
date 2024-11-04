import os
import json
import yaml
from datetime import datetime

def create_project_structure():
  # إنشاء المجلدات الرئيسية
  folders = [
      'documents/تقنية',
      'documents/علوم',
      'documents/أعمال',
      'documents/تعليم',
      'cache',
      'logs'
  ]
  
  for folder in folders:
      os.makedirs(folder, exist_ok=True)
      print(f"تم إنشاء المجلد: {folder}")

  # إنشاء ملف الإعدادات
  config = {
      'document_settings': {
          'allowed_extensions': ['.txt', '.md'],
          'max_file_size': 5242880
      },
      'categories': ['تقنية', 'علوم', 'أعمال', 'تعليم'],
      'metadata_fields': {
          'required': ['title', 'category', 'date', 'keywords'],
          'optional': ['source', 'author', 'version']
      },
      'search_settings': {
          'case_sensitive': False,
          'include_metadata': True,
          'max_results': 50
      }
  }
  
  with open('config.yaml', 'w', encoding='utf-8') as f:
      yaml.dump(config, f, allow_unicode=True)
  print("تم إنشاء ملف الإعدادات")

  # إنشاء ملف metadata.json فارغ
  with open('documents/metadata.json', 'w', encoding='utf-8') as f:
      json.dump({}, f, ensure_ascii=False, indent=4)
  print("تم إنشاء ملف البيانات الوصفية")

  return "تم إنشاء هيكل المشروع بنجاح"

# إنشاء مستندات نموذجية
def create_sample_documents():
  documents = [
      {
          'title': 'مقدمة في الذكاء الاصطناعي',
          'category': 'تقنية',
          'content': """
          الذكاء الاصطناعي هو فرع من فروع علوم الحاسوب يهتم بتطوير أنظمة ذكية
          تحاكي القدرات البشرية في التفكير والتعلم واتخاذ القرارات.
          
          يشمل الذكاء الاصطناعي عدة مجالات منها:
          1. التعلم الآلي
          2. معالجة اللغات الطبيعية
          3. الرؤية الحاسوبية
          4. الروبوتات
          
          تطبيقات الذكاء الاصطناعي متنوعة وتشمل:
          - المساعدات الصوتية
          - أنظمة التوصية
          - السيارات ذاتية القيادة
          - التشخيص الطبي
          """,
          'keywords': ['ذكاء اصطناعي', 'تعلم آلي', 'تقنية']
      },
      {
          'title': 'أساسيات البرمجة',
          'category': 'تعليم',
          'content': """
          البرمجة هي عملية كتابة التعليمات للحاسوب لتنفيذ مهام محددة.
          
          المفاهيم الأساسية في البرمجة:
          1. المتغيرات
          2. الهياكل الشرطية
          3. الحلقات التكرارية
          4. الدوال
          
          لغات البرمجة الشائعة:
          - Python
          - JavaScript
          - Java
          - C++
          """,
          'keywords': ['برمجة', 'تعليم', 'لغات برمجة']
      }
  ]
  
  metadata = {}
  
  for doc in documents:
      filename = f"{doc['title']}_{datetime.now().strftime('%Y%m%d')}.txt"
      filepath = os.path.join('documents', doc['category'], filename)
      
      # إنشاء محتوى المستند
      content = f"""العنوان: {doc['title']}
التصنيف: {doc['category']}
التاريخ: {datetime.now().strftime('%Y-%m-%d')}
الكلمات المفتاحية: {', '.join(doc['keywords'])}
---
{doc['content']}
"""
      
      # حفظ المستند
      with open(filepath, 'w', encoding='utf-8') as f:
          f.write(content)
      
      # إضافة البيانات الوصفية
      metadata[filename] = {
          'title': doc['title'],
          'category': doc['category'],
          'date': datetime.now().strftime('%Y-%m-%d'),
          'keywords': doc['keywords']
      }
      
      print(f"تم إنشاء المستند: {filename}")
  
  # تحديث ملف البيانات الوصفية
  with open('documents/metadata.json', 'w', encoding='utf-8') as f:
      json.dump(metadata, f, ensure_ascii=False, indent=4)

# إنشاء ملف requirements.txt
def create_requirements():
  requirements = """langchain==0.0.184
chromadb==0.3.22
openai==0.27.8
tiktoken==0.4.0
unstructured==0.7.12
pyyaml==6.0
python-dotenv==1.0.0"""
  
  with open('requirements.txt', 'w') as f:
      f.write(requirements)
  print("تم إنشاء ملف requirements.txt")

# إنشاء ملف .env
def create_env():
  env_content = """# OpenAI API Key
OPENAI_API_KEY=your-api-key-here

# Other Configuration
MAX_TOKENS=2000
TEMPERATURE=0
"""
  with open('.env', 'w') as f:
      f.write(env_content)
  print("تم إنشاء ملف .env")

# تشغيل كل الوظائف
def setup_project():
  print("بدء إعداد المشروع...")
  create_project_structure()
  create_sample_documents()
  create_requirements()
  create_env()
  print("\nتم إعداد المشروع بنجاح!")
  print("\nالخطوات التالية:")
  print("1. قم بتثبيت المتطلبات باستخدام: pip install -r requirements.txt")
  print("2. أضف مفتاح API الخاص بك في ملف .env")
  print("3. أضف المستندات الخاصة بك في المجلدات المناسبة")

# تشغيل الإعداد
if __name__ == "__main__":
  setup_project()