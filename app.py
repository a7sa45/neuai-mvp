import time
import uuid
import imghdr
import os
import numpy as np
from six import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
from flask import Flask, render_template, flash, request, redirect, url_for, abort, send_from_directory
from werkzeug.utils import secure_filename
from object_detection.utils import label_map_util
from keras.models import load_model
import cv2

#-----------------------------------------Load the model--------------------------------
start_time = time.time()
tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load('static/my_model/saved_model')
end_time = time.time()
elapsed_time = end_time - start_time
print('Elapsed time: ' + str(elapsed_time) + 's')


#----------------------------------Function to convert image into numpy array-----------------
def load_image_into_numpy_array(path):
        return np.array(Image.open(path))


# -----------------------------------------Load the Label Map--------------------------------
category_index = {
        1: {'id': 1, 'name': 'windsheld'},
}

def page_not_found(e):
  return render_template('404.html'), 404

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'
app.register_error_handler(404, page_not_found)

def validate_image(stream):
    header = stream.read(512)
    stream.seek(0) 
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

@app.errorhandler(413)
def too_large(e):
    return render_template('404.html'), 404
@app.errorhandler(400)
def too_large(e):
    return render_template('404.html'), 404



@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('index.html', files=files)

@app.route('/', methods=['POST'])
def upload_files():
    #uploaded_file = request.form['file']

    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    # ----------------------------------------- end upload -------------------------------- 

    # ----------------------------------------- Start Model Processing --------------------------------
    image_path = "uploads/" + filename

    # -------------------------------------------Step 1 convert img to np array -----------------------
    image_np = load_image_into_numpy_array(image_path)

    # ------------------------------------------- git size of image -----------------------
    img_data = tf.io.gfile.GFile(image_path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    #print(im_width)

    input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    plt.rcParams['figure.figsize'] = [42, 21]
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=18,
        max_boxes_to_draw=200,
        min_score_thresh=.40,
        agnostic_mode=False)

    # ------------------------------------------- save image befor crop -----------------------
    image_None = Image.fromarray(image_np_with_detections)
    path_none_image = 'none_{}.jpg'.format(str(uuid.uuid1()))
    image_None.save('static/output/' + path_none_image)


    #plt.subplot(4, 1, 1)
    #plt.imshow(image_np_with_detections)
    
    # -------------------------------------------Step 2 crop the images -----------------------
    x = detections['detection_boxes'][0].numpy()
    # num_detections = int(detections.pop('num_detections'))
    z = np.array(x[0])
    #im_width, im_height
    b = np.array([im_height, im_width, im_height, im_width])
    a = z * b
    a = a.astype(int)

    image_path = image_np_with_detections
    # Load image
    # image = Image.open(image_path)
    
    # Convert image to array
    image_arr = np.array(image_np_with_detections)
    
    # Crop image
    image_arr = image_arr[a[0]:a[2], a[1]:a[3]]

    # Convert array to image
    image = Image.fromarray(image_arr)

    # -------------------------------------------Step 3 Save croped image ----------------------
    c = r'uploads/'
    image.save(c + 'windsheld__{}.JPG'.format(str(uuid.uuid1())))

    #-----------------------------------Second model---------------------------------------------
    
    model = load_model('static/second_model/model.h5')

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    #img = cv2.imread(name)
    img = cv2.resize(image_arr,(250,250))
    img = np.reshape(img,[1,250,250,3])

    x = model.predict(img)
    y_classes = x.argmax()
    
    if y_classes==2:
        color = "border-success"
    else:
        color = "border-danger"
  

    return render_template('index.html', image = path_none_image, classs = y_classes, color = color)
    #return redirect(url_for('index', image = path_none_image))

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)


if __name__ == '__main__':
    app.run(debug=True)