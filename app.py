from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_talisman import Talisman
import os
from bot import DocumentBot
import logging
from datetime import datetime
import traceback

# Initialize Flask app
app = Flask(__name__)
CORS(app)
Talisman(app, force_https=False)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize bot
try:
    bot = DocumentBot()
    bot_status = "ready"
    logger.info("تم تحميل البوت بنجاح")
except Exception as e:
    bot = None
    bot_status = str(e)
    logger.error(f"خطأ في تحميل البوت: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handle both page render and questions"""
    try:
        # Handle question from URL parameter or form
        question = request.args.get('question') or request.form.get('question')
        
        if question:
            logger.info(f"معالجة السؤال: {question}")
            
            if not bot:
                return render_template('index.html', 
                    bot_status=bot_status,
                    error="البوت غير متاح حالياً"
                )
            
            try:
                response = bot.answer_question(question)
                logger.info("تم إنشاء الإجابة")
                return render_template('index.html',
                    bot_status=bot_status,
                    question=question,
                    answer=response["answer"],
                    sources=response["sources"]
                )
            except Exception as e:
                logger.error(f"خطأ في معالجة السؤال: {str(e)}")
                logger.error(traceback.format_exc())
                return render_template('index.html',
                    bot_status=bot_status,
                    error=f"حدث خطأ في معالجة السؤال: {str(e)}"
                )
        
        # Just render the page if no question
        return render_template('index.html', bot_status=bot_status)
        
    except Exception as e:
        logger.error(f"خطأ عام: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template('index.html',
            bot_status="error",
            error=f"حدث خطأ: {str(e)}"
        )

@app.route('/api/ask', methods=['POST'])
def ask():
    """API endpoint for questions"""
    try:
        if not bot:
            return jsonify({
                "error": "البوت غير متاح حالياً",
                "details": bot_status
            }), 503

        question = request.form.get('question', '').strip()
        if not question:
            return jsonify({
                "error": "الرجاء إدخال سؤال"
            }), 400

        logger.info(f"معالجة السؤال عبر API: {question}")
        response = bot.answer_question(question)
        logger.info("تم إنشاء الإجابة")
        
        return jsonify({
            "answer": response["answer"],
            "sources": response["sources"]
        })

    except Exception as e:
        logger.error(f"خطأ في API: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": f"حدث خطأ في معالجة السؤال: {str(e)}"
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ready' if bot else 'error',
        'bot_status': bot_status,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
