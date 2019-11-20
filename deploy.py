from PIL import Image
import json
import numpy as np
import tensorflow.keras as keras
from werkzeug.wrappers import Request, Response

model = None

def load_image(environ):
    request = Request(environ)
    if not request.files:
        return None
    file_key = list(request.files.keys())[0]
    file = request.files.get(file_key)

    img = Image.open(file.stream)
    img.load()
    return img
    
def preprocess_image(img):
    img = img.resize((28, 28)).convert('L')
    image_data = np.reshape(np.array(img, dtype='float32'), (1, 28, 28))
    max_value = np.max(image_data)
    if max_value != 0:
        image_data /= max_value
    return image_data

def application(environ, start_response):
    img = load_image(environ)
    if not img:
        return Response('no file uploaded', 400)(environ, start_response)
    image_data = preprocess_image(img)

    global model
    if not model:
        model = keras.models.load_model('model/mnist.h5')

    prediction = model.predict(image_data)
    result = {}
    for i in range(0, 10):
        result[str(i)] = str(prediction[0][i])

    response = Response(json.dumps(result), mimetype='application/json')
    return response(environ, start_response)

if __name__ == "__main__":
    from werkzeug.serving import run_simple
    run_simple('localhost', 8000, application)
