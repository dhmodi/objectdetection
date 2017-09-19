#!/usr/bin/env python

from __future__ import print_function
from future.standard_library import install_aliases

install_aliases()


import os
from flask import Flask, request, redirect, url_for, flash, session, render_template, jsonify
from flask_session import Session
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import warnings
warnings.simplefilter("ignore", UserWarning)
from ssd.ssd import SSD300
from ssd.ssd_utils import BBoxUtility
import cv2
import numpy as np
import gc
import json
gc.collect()
from flask import Response


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
sess = Session()
sess.init_app(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def imageUpload():
    flash('Image Upload')
    return render_template('dashboard.html', page_title='My Page!')


@app.route('/image', methods=['POST'])
def detect_file():
    print("Inside Upload")
    category = "Others"
    filename = ""
    output = []
    if request.method == 'POST':
        # check if the post request has the file part
        print(len(request.files))
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        print(file)
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        print(file.filename)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('upload_file',
            #                         filename=filename))
            np.set_printoptions(suppress=True)
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.45
            set_session(tf.Session(config=config))

            voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                           'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                           'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
                           'Sheep', 'Sofa', 'Train', 'Tvmonitor']
            NUM_CLASSES = len(voc_classes) + 1
            inputs = []
            input_shape=(300, 300, 3)
            model = SSD300(input_shape, num_classes=NUM_CLASSES)
            model.load_weights('weight/weights_SSD300.hdf5', by_name=True)
            bbox_util = BBoxUtility(NUM_CLASSES)
            img_path = UPLOAD_FOLDER + "/" + filename
            img = image.load_img(img_path, target_size=(300, 300))
            img = image.img_to_array(img)
            inputs.append(img.copy())
            inputs = preprocess_input(np.array(inputs))

            preds = model.predict(inputs, batch_size=1, verbose=1)
            results = bbox_util.detection_out(preds)
            det_label = results[0][:, 0]
            det_conf = results[0][:, 1]
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
            top_label_indices = det_label[top_indices].tolist()
            top_conf = det_conf[top_indices]
            for i in range(top_conf.shape[0]):
                score = top_conf[i]
                label = int(top_label_indices[i])
                label_name = voc_classes[label - 1]
                print(str(score))
                output.append("{'category':" + label_name + ", 'probability':" + str(score) + "}")
            #
    return  jsonify(fileName=filename, results=output)

@app.route('/htmlimage', methods=['POST'])
def upload_file():
    print("Inside HTML Upload")
    category = "Others"
    filename = ""
    output = []
    if request.method == 'POST':
        # check if the post request has the file part
        print(len(request.files))
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        print(file)
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        print(file.filename)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('upload_file',
            #                         filename=filename))
            np.set_printoptions(suppress=True)
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.45
            set_session(tf.Session(config=config))

            voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                           'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                           'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
                           'Sheep', 'Sofa', 'Train', 'Tvmonitor']
            NUM_CLASSES = len(voc_classes) + 1
            inputs = []
            input_shape=(300, 300, 3)
            model = SSD300(input_shape, num_classes=NUM_CLASSES)
            model.load_weights('weight/weights_SSD300.hdf5', by_name=True)
            bbox_util = BBoxUtility(NUM_CLASSES)
            img_path = UPLOAD_FOLDER + "/" + filename
            img = image.load_img(img_path, target_size=(300, 300))
            img = image.img_to_array(img)
            inputs.append(img.copy())
            inputs = preprocess_input(np.array(inputs))

            preds = model.predict(inputs, batch_size=1, verbose=1)
            results = bbox_util.detection_out(preds)
            det_label = results[0][:, 0]
            det_conf = results[0][:, 1]
            det_xmin = results[0][:, 2]
            det_ymin = results[0][:, 3]
            det_xmax = results[0][:, 4]
            det_ymax = results[0][:, 5]
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
            top_label_indices = det_label[top_indices].tolist()
            top_conf = det_conf[top_indices]
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]

            for i in range(top_conf.shape[0]):
                xmin = int(round(top_xmin[i] * img.shape[1]))
                ymin = int(round(top_ymin[i] * img.shape[0]))
                xmax = int(round(top_xmax[i] * img.shape[1]))
                ymax = int(round(top_ymax[i] * img.shape[0]))
                score = top_conf[i] * 100
                label = int(top_label_indices[i])
                label_name = voc_classes[label - 1]
                print("Object" + label_name + ":" + str(round(score,2)))
                output.append('{ "category": "' + label_name + '", "probability": "' + str(round(score,2)) + '%", "xmin": "' + str(xmin) + '", "ymin": "' + str(ymin) + '", "xmax": "' + str(xmax) + '", "ymax": "' + str(ymax) + '"}')
                #
    return  jsonify(fileName=filename, results=output)

if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    port = int(os.getenv('PORT', 5000))
    # static = Environment(app)
    #
    # js = Bundle('jquery.js', 'base.js', 'widgets.js',
    #         filters='jsmin', output='gen/packed.js')
    # static.register('js_all', js)
    print("Starting app on port %d" % port)

    app.run(debug=True, port=port, host='0.0.0.0')