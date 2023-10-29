from flask import Flask, request, jsonify
import model.inference as infer
import mysql.connector

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'webp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 분류되는 음식 category들을 전부 반환
@app.route('/food/list', methods=["GET"])
def get_food_categories():
    categories = infer.get_classes()
    return jsonify(categories)


# 음식 이미지를 입력받아 상위 5개의 음식을 반환
@app.route('/food', methods=["POST"])
def inference_image():
    if "image" not in request.files:
        return "Image not received", 400
    image_file = request.files['image']
    if not image_file or not allowed_file(image_file.filename):
        return "Invalid image file", 400
    foods = infer.inference(image_file)

    return jsonify(result(foods))


def result(foods):
    result = {"food": []}
    for food in foods:
        id = find_by_name(food)
        if id is not None:
            id = id[0]
        result["food"].append({
            "id": id,
            "name": food
        })
    return result


def find_by_name(food_name):
    cursor = mysql.cursor(buffered=True)
    print(food_name)
    cursor.execute("SELECT food.food_id FROM food WHERE food.food_name = %s", (food_name,))
    data = cursor.fetchone()
    cursor.close()
    return data


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
