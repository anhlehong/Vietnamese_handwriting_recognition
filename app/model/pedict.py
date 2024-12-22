from app.utils.image_processing import preprocess_image
from app.model.vnocr import predict_line

def predict(model, char_list, image_path):
    # print(image_path)
    image = preprocess_image(image_path)
    line = predict_line(model, char_list, image)
    return line