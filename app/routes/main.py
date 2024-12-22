import os
import shutil
from flask import Blueprint, render_template, request, current_app, jsonify
from app.services.text_extraction import extract_lines, process_and_invert_image
from app.model.pedict import predict
import cv2
from dotenv import load_dotenv
import requests

main = Blueprint('main', __name__)

env_path = os.path.join(os.path.dirname(__file__), '.env.local')
print("env_path: ", env_path)

# Tải các biến môi trường từ tệp .env.local
load_dotenv(dotenv_path=env_path)

# Truy cập các biến môi trường
API_KEY = os.getenv('API_KEY')
GEMINI_API_URL = os.getenv('GEMINI_API_URL')
debug_mode = os.getenv('DEBUG')
print("API_KEY: ", API_KEY)

@main.route('/')
def home():
    # Lấy các thư mục cần xử lý
    processed_folder = current_app.config['PROCESSED_FOLDER']
    line_folder = current_app.config['LINE_FOLDER']
    upload_folder = current_app.config['UPLOAD_FOLDER']

    # Hàm xóa tất cả file trong thư mục
    def clear_folder(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)  # Xóa file
            except Exception as e:
                print(f"Không thể xóa file {file_path}: {e}")

    # Hàm xóa toàn bộ thư mục và tạo lại
    def clear_and_recreate_folder(folder):
        try:
            if os.path.exists(folder):
                shutil.rmtree(folder)  # Xóa toàn bộ thư mục
            os.makedirs(folder, exist_ok=True)  # Tạo lại thư mục trống
        except Exception as e:
            print(f"Không thể xóa thư mục {folder}: {e}")

    # Xóa file trong các thư mục processed và line
    clear_folder(processed_folder)
    clear_folder(line_folder)
    # Xóa toàn bộ thư mục uploads và tạo lại
    clear_and_recreate_folder(upload_folder)

    return render_template('upload.html')


@main.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Các thư mục
    upload_folder = current_app.config['UPLOAD_FOLDER']
    processed_folder = current_app.config['PROCESSED_FOLDER']
    line_folder = current_app.config['LINE_FOLDER']

    # Đảm bảo các thư mục tồn tại
    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)
    os.makedirs(line_folder, exist_ok=True)

    # Lưu file gốc
    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)

    # 1. Tiền xử lý và đảo màu, lưu vào thư mục "processed"
    image_invert_path = process_and_invert_image(filepath, processed_folder)

    # 2. Đọc lại ảnh đã xử lý
    inverted_image = cv2.imread(image_invert_path)

    # 3. Tách các dòng văn bản, lưu vào thư mục "line"
    processed_images, lines = extract_lines(inverted_image, line_folder)

    # Sử dụng model đã tải trong app context
    model = current_app.config['MODEL']
    char_list = current_app.config['CHAR_LIST']

    texts = []
    for line in lines:
        texts.append(predict(model, char_list, line))
    print("texts: ", texts)
    
    content = "; ".join(texts)
    prompt = (
        f"Sửa lại nội dung sau cho đúng ngữ nghĩa, ngữ pháp tiếng Việt. với dấu ; là kết thúc mỗi câu, yêu cầu đầy đủ nội dung."
        f"Viết lại cho hoàn chỉnh cấu trúc file markdown. Không nói gì thêm. "
        f"Nội dung là: {content}"
    )

    print("prompt: ", prompt)
    if not prompt:
        return jsonify({"error": "Câu hỏi là bắt buộc"}), 400

    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }
    print(headers)
    print(payload)
    print(f"{os.getenv('GEMINI_API_URL')}?key={os.getenv('API_KEY')}")
    print(API_KEY)
    print(GEMINI_API_URL)
    try:
        response = requests.post(
            f"{os.getenv('GEMINI_API_URL')}?key={os.getenv('API_KEY')}",
            headers=headers,
            json=payload
        )
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        data = response.json()
        print("data: ", data)
        
        # Tạo đường dẫn URL cho ảnh đã xử lý và các dòng văn bản
        processed_image_url = f"static/processed/{os.path.basename(image_invert_path)}"
        line_urls = [
            f"static/line/{os.path.basename(img)}" for img in processed_images
        ]

        # Extract the markdown content from the response
        markdown_content = data['candidates'][0]['content']['parts'][0]['text']
        
        # Loại bỏ ký tự không mong muốn ở đầu và cuối
        if markdown_content.startswith("```markdown"):
            markdown_content = markdown_content[len("```markdown"):].strip()
        if markdown_content.endswith("```"):
            markdown_content = markdown_content[:-len("```")].strip()
        print("markdown_content: \n", markdown_content)

        return render_template('display.html', markdown_content=markdown_content)
            
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

