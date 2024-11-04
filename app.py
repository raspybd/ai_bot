from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_talisman import Talisman
import os
from bot import DocumentBot
import logging
from datetime import datetime
import time
from typing import Dict, Any
from functools import wraps
import yaml
import codecs

# Initialize Flask app
app = Flask(__name__)
CORS(app)
Talisman(app)

# Load configuration
with codecs.open('config.yaml', 'r', 'utf-8') as f:
    config = yaml.safe_load(f)

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename=f'logs/app_{datetime.now().strftime("%Y%m%d")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# Initialize cache
response_cache: Dict[str, Any] = {}
CACHE_DURATION = 3600  # 1 hour

def cache_response(func):
    """Decorator to cache responses"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = str(request.form.get('question', ''))
        current_time = time.time()
        
        # Check if response is in cache and not expired
        if cache_key in response_cache:
            cached_response, timestamp = response_cache[cache_key]
            if current_time - timestamp < CACHE_DURATION:
                logging.info(f"استجابة من الذاكرة المؤقتة: {cache_key}")
                return cached_response
        
        # Get new response
        response = func(*args, **kwargs)
        
        # Cache the response
        response_cache[cache_key] = (response, current_time)
        
        # Clean old cache entries
        clean_cache()
        
        return response
    return wrapper

def clean_cache():
    """Remove expired cache entries"""
    current_time = time.time()
    expired_keys = [
        k for k, (_, timestamp) in response_cache.items()
        if current_time - timestamp > CACHE_DURATION
    ]
    for k in expired_keys:
        del response_cache[k]

def rate_limit_check(max_requests: int = 100, time_window: int = 3600) -> bool:
    """Check if request is within rate limits"""
    client_ip = request.remote_addr
    current_time = time.time()
    
    if not hasattr(app, 'rate_limit_data'):
        app.rate_limit_data = {}
    
    # Clean old rate limit data
    app.rate_limit_data = {
        ip: (count, timestamp) 
        for ip, (count, timestamp) in app.rate_limit_data.items()
        if current_time - timestamp < time_window
    }
    
    # Check current IP
    if client_ip in app.rate_limit_data:
        count, timestamp = app.rate_limit_data[client_ip]
        if current_time - timestamp < time_window:
            if count >= max_requests:
                return False
            app.rate_limit_data[client_ip] = (count + 1, timestamp)
    else:
        app.rate_limit_data[client_ip] = (1, current_time)
    
    return True

class APIError(Exception):
    """Custom API error class"""
    def __init__(self, message, status_code=400):
        super().__init__()
        self.message = message
        self.status_code = status_code

@app.errorhandler(APIError)
def handle_api_error(error):
    """Handle API errors"""
    response = jsonify({'error': error.message})
    response.status_code = error.status_code
    return response

# Initialize bot
try:
    bot = DocumentBot()
    logging.info("تم تحميل البوت بنجاح")
except Exception as e:
    logging.error(f"خطأ في تحميل البوت: {str(e)}")
    bot = None

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
@cache_response
def ask():
    """Handle questions via API"""
    try:
        if not rate_limit_check():
            raise APIError("تم تجاوز حد الطلبات المسموح به", 429)
        
        if not bot:
            raise APIError("البوت غير متاح حالياً", 503)
            
        question = request.form.get('question', '').strip()
        if not question:
            raise APIError("الرجاء إدخال سؤال", 400)
            
        logging.info(f"سؤال جديد: {question}")
        
        response = bot.answer_question(question)
        logging.info("تم إرسال الإجابة بنجاح")
        
        return jsonify(response)
        
    except APIError as e:
        raise
    except Exception as e:
        logging.error(f"خطأ غير متوقع: {str(e)}")
        raise APIError("حدث خطأ في معالجة طلبك", 500)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if bot else 'unavailable',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

# Create a basic HTML template
@app.route('/templates/index.html')
def serve_template():
    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
