# check_setup.py
import os
import sys

def check_project_setup():
  # قائمة بالملفات والمجلدات المطلوبة
  required_structure = {
      'directories': ['documents', 'cache', 'logs'],
      'files': ['config.yaml', '.env', 'requirements.txt', 'bot.py']
  }
  
  # التحقق من المجلدات
  print("\nالتحقق من المجلدات:")
  for directory in required_structure['directories']:
      if os.path.isdir(directory):
          print(f"✅ {directory} موجود")
      else:
          print(f"❌ {directory} غير موجود")
  
  # التحقق من الملفات
  print("\nالتحقق من الملفات:")
  for file in required_structure['files']:
      if os.path.isfile(file):
          print(f"✅ {file} موجود")
      else:
          print(f"❌ {file} غير موجود")
  
  # التحقق من المتطلبات
  print("\nالتحقق من المتطلبات:")
  try:
      with open('requirements.txt', 'r') as f:
          requirements = f.read().strip().split('\n')
      import pkg_resources
      installed = [pkg.key for pkg in pkg_resources.working_set]
      missing = []
      for requirement in requirements:
          package = requirement.split('==')[0]
          if package.lower() not in [pkg.lower() for pkg in installed]:
              missing.append(package)
      
      if missing:
          print("❌ بعض المتطلبات غير مثبتة:")
          for package in missing:
              print(f"   - {package}")
      else:
          print("✅ جميع المتطلبات مثبتة")
  except Exception as e:
      print(f"❌ خطأ في التحقق من المتطلبات: {str(e)}")
  
  # التحقق من ملف .env
  print("\nالتحقق من ملف .env:")
  try:
      with open('.env', 'r') as f:
          env_content = f.read()
      if 'OPENAI_API_KEY' in env_content:
          print("✅ OPENAI_API_KEY موجود في ملف .env")
      else:
          print("❌ OPENAI_API_KEY غير موجود في ملف .env")
  except:
      print("❌ خطأ في قراءة ملف .env")

if __name__ == "__main__":
  print("جاري التحقق من إعداد المشروع...")
  check_project_setup()