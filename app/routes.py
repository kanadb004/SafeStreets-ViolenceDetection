import os
from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename
from .utils import allowed_file
from .ml_model.model import predict_video

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/about')
def about():
    return render_template('about.html')

@main.route('/contact')
def contact():
    return render_template('contact.html')

@main.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if video and allowed_file(video.filename):
        filename = secure_filename(video.filename)
        save_path = os.path.join('footages', filename)
        video.save(save_path)
        result = predict_video(save_path)
        return jsonify({'message': 'File uploaded successfully', 'prediction': result})
    return jsonify({'error': 'Invalid file type'}), 400
