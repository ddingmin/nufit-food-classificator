from flask import Flask, request, jsonify
import json
import model.inference as infer

app = Flask(__name__)

# 분류되는 음식 category들을 전부 반환
@app.route('/food/list', methods=["GET"])
def get_food_categories():
    categories = infer.get_classes()
    return jsonify(categories)

# 음식 이미지를 입력받아 상위 5개의 음식을 반환
@app.route('/food', methods=["GET"])
def inference_image():
    if "image" in request.files:
        image_file = request.files['image']
        predictions = infer.inference(image_file)
        return jsonify(predictions)
    else:
        return "Image not received", 400

if __name__ == '__main__':
    app.run()