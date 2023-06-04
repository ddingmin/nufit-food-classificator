from flask import Flask, request, jsonify
import model.inference as infer

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','webp'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 분류되는 음식 category들을 전부 반환
@app.route('/food/list', methods=["GET"])
def get_food_categories():
    categories = infer.get_classes()
    return jsonify(categories)

# 음식 이미지를 입력받아 상위 5개의 음식을 반환
@app.route('/food', methods=["GET"])
def inference_image():
    if "image" not in request.files:
        return "Image not received", 400
    image_file = request.files['image']
    if not image_file or not allowed_file(image_file.filename):
        return "Invalid image file", 400
    predictions = infer.inference(image_file)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run()