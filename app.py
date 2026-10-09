"""
app.py
------
Flask Web Application & REST API entry point for Garbage Classification System.

Exposes web UI routes and REST API endpoints:
- GET  /                           : Main glassmorphic Web UI dashboard
- POST /api/predict                : Image file upload classification endpoint
- GET  /api/sample_predict/<cls>   : Instant test prediction endpoint using dataset samples
- GET  /api/health                 : API health & model status check
- GET  /api/stats                  : Dataset & classification metrics summary
"""

import os
import random
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

from src.predict import predict_image, CLASSES, RECYCLING_GUIDANCE

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Renders main Web UI dashboard."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API Endpoint for uploading garbage item image and receiving CNN classification results.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded in request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image file'}), 400

    if file and allowed_file(file.filename):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            result = predict_image(filepath)
            result['image_url'] = f"/static/uploads/{filename}"
            return jsonify(result), 200
        except Exception as e:
            return jsonify({'error': f'Prediction execution error: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file extension. Supported: .jpg, .jpeg, .png, .webp'}), 400


@app.route('/api/sample_predict/<category>', methods=['GET'])
def api_sample_predict(category: str):
    """
    API Endpoint for testing prediction on a random dataset sample for the given class category.
    """
    category_lower = category.lower()
    if category_lower not in CLASSES:
        return jsonify({'error': f'Unknown category "{category}". Valid: {CLASSES}'}), 400

    # Search for sample image in dataset
    dataset_base = Path("dataset")
    possible_dirs = [
        dataset_base / category_lower,
        dataset_base / "dataset-resized" / category_lower
    ]

    sample_files = []
    for d in possible_dirs:
        if d.is_dir():
            found = [f for f in d.glob("*.*") if f.suffix.lower() in {'.jpg', '.png', '.jpeg'}]
            sample_files.extend(found)

    if not sample_files:
        return jsonify({'error': f'No dataset samples found for category "{category}".'}), 404

    sample_file = random.choice(sample_files)

    try:
        result = predict_image(str(sample_file))
        # Copy to uploads so static web server can render preview
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = f"sample_{category_lower}_{sample_file.name}"
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        import shutil
        shutil.copy(sample_file, target_path)

        result['sample_image_url'] = f"/static/uploads/{filename}"
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': f'Sample prediction error: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint."""
    model_exists = os.path.exists("models/best_garbage_cnn.keras")
    return jsonify({
        'status': 'healthy',
        'model_trained': model_exists,
        'model_path': 'models/best_garbage_cnn.keras',
        'classes': CLASSES
    }), 200


@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Returns classification categories and system stats."""
    return jsonify({
        'target_classes': CLASSES,
        'class_count': len(CLASSES),
        'recycling_guidance': RECYCLING_GUIDANCE
    }), 200


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    print("========================================")
    print("STARTING GARBAGE CLASSIFIER WEB APP")
    print("========================================")
    print("Access UI at: http://localhost:5000")
    print("========================================\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
