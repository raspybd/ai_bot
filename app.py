from flask import Flask, render_template, request
from flask_cors import CORS
#from flask_talisman import Talisman
import os
from bot import DocumentBot

app = Flask(__name__)
CORS(app)
#Talisman(app)

try:
  bot = DocumentBot()
  print("تم تحميل البوت بنجاح!")
except Exception as e:
  print(f"خطأ في تحميل البوت: {str(e)}")

@app.route("/", methods=["GET", "POST"])
def index():
  try:
      if request.method == "POST":
          question = request.form["question"]
          response = bot.answer_question(question)
          return render_template(
              "index.html",
              answer=response["answer"],
              sources=response["sources"]
          )
      return render_template("index.html")
  except Exception as e:
      return str(e), 500

if __name__ == "__main__":
  port = int(os.environ.get("PORT", 8080))
  app.run(host='0.0.0.0', port=port)
