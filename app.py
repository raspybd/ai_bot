from flask import Flask, render_template, request, jsonify, send_from_directory
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

# Disable HTTPS requirement during development
Talisman(app, force_https=False)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize bot
try:
    bot = DocumentBot()
    logger.info("Bot loaded successfully")
except Exception as e:
    logger.error(f"Error loading bot: {str(e)}")
    logger.error(traceback.format_exc())
    bot = None

@app.route('/')
def index():
    """Render main page"""
    try:
        logger.info("Accessing index page")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}")
        logger.error(traceback.format_exc())
        return str(e), 500

@app.route('/api/ask', methods=['GET', 'POST'])
def ask():
    """Handle questions"""
    try:
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request data: {request.form}")
        
        if not bot:
            return jsonify({
                "error": "Bot is not initialized",
                "details": "Please check server logs"
            }), 503

        if request.method == 'POST':
            question = request.form.get('question', '')
        else:
            question = request.args.get('question', '')

        logger.info(f"Received question: {question}")

        if not question:
            return jsonify({
                "error": "No question provided",
                "details": "Please provide a question"
            }), 400

        response = bot.answer_question(question)
        logger.info("Answer generated successfully")
        
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy' if bot else 'bot_not_initialized',
            'timestamp': datetime.now().isoformat(),
            'bot_status': 'loaded' if bot else 'not_loaded'
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
