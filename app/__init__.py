from flask import Flask
from app.routes.main import main
import os


def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'app/static/uploads' 
    app.config['PROCESSED_FOLDER'] = 'app/static/processed'
    app.config['LINE_FOLDER'] = 'app/static/line'  

    # Đảm bảo các thư mục tồn tại
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    os.makedirs(app.config['LINE_FOLDER'], exist_ok=True)

    app.register_blueprint(main)
    return app
