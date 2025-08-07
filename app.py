from flask import Flask, render_template, request
from ultralytics import YOLO
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model_path = os.path.join(os.path.dirname(__file__), "best.pt")
assert os.path.exists(model_path), f"模型路徑不存在: {model_path}"

print("載入模型中...")
model = YOLO(model_path)
print("模型載入成功")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    resultpath = os.path.join(RESULT_FOLDER, filename)

    file.save(filepath)

    results = model.predict(source=filepath, save=False)
    im_array = results[0].plot()
    im = Image.fromarray(im_array)
    im.save(resultpath)

    num_objects = len(results[0].boxes)
    result_image = f"/static/results/{filename}"

    return render_template('index.html', result_image=result_image, num_objects=num_objects)

if __name__ == '__main__':
    app.run()

































