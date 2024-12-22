from flask import Flask
from app.model.vnocr import load_model

def create_app():
    app = Flask(__name__)

    # Cấu hình các thư mục
    app.config['UPLOAD_FOLDER'] = './uploads'
    app.config['PROCESSED_FOLDER'] = './static/processed'
    app.config['LINE_FOLDER'] = './static/line'
    app.config['MODEL_FOLDER'] = './app/model'

    # Tải model chỉ một lần khi ứng dụng khởi chạy
    print("Loading model...")
    model_folder = app.config['MODEL_FOLDER']
    model, char_list = load_model(model_folder)
    app.config['MODEL'] = model
    app.config['CHAR_LIST'] = char_list
    print("Model loaded successfully!")

    # Đăng ký blueprint
    from app.routes.main import main
    app.register_blueprint(main)

    return app
