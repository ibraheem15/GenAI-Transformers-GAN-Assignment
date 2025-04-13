from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import base64
from PIL import Image
import io
from tensorflow.keras import layers

# Add the custom InstanceNormalization layer with proper registration
class InstanceNormalization(layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        depth = input_shape[-1]
        self.scale = self.add_weight(
            name='scale',
            shape=[depth],
            initializer=tf.random_normal_initializer(1.0, 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=[depth],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({
            'epsilon': self.epsilon,
        })
        return config

app = Flask(__name__)

# Load the models with custom_objects
MODEL_DIR = "saved_models"
custom_objects = {'InstanceNormalization': InstanceNormalization}
generator_g = load_model(os.path.join(MODEL_DIR, "generator_g.keras"), custom_objects=custom_objects)  # sketch to photo
generator_f = load_model(os.path.join(MODEL_DIR, "generator_f.keras"), custom_objects=custom_objects)  # photo to sketch

def normalize_image(image):
    """Normalize image to [-1, 1] range"""
    image = tf.cast(image, tf.float32)
    return (image / 127.5) - 1

def preprocess_image(image_data):
    """Preprocess image for model input"""
    # Convert to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Resize to 256x256
    img = cv2.resize(img, (256, 256))
    
    # Normalize
    img = normalize_image(img)
    
    # Add batch dimension
    img = tf.expand_dims(img, 0)
    return img

def is_sketch(image_data):
    """Detect if image is a sketch using simple edge detection heuristic"""
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Calculate the percentage of edge pixels
    edges = cv2.Canny(img, 100, 200)
    edge_percentage = np.count_nonzero(edges) / edges.size
    
    # If edge percentage is high, likely a sketch
    return edge_percentage > 0.1

def process_image(image_data):
    """Process image through appropriate generator"""
    # Determine if input is sketch or photo
    if is_sketch(image_data):
        generator = generator_f  # photo to sketch
    else:
        generator = generator_g  # sketch to photo
    
    # Preprocess image
    input_image = preprocess_image(image_data)
    
    # Generate output
    output = generator(input_image)
    
    # Convert output tensor to image
    output_image = (output[0] * 0.5 + 0.5).numpy()
    output_image = (output_image * 255).astype(np.uint8)
    
    # Convert to base64 for display
    output_pil = Image.fromarray(output_image)
    output_buffer = io.BytesIO()
    output_pil.save(output_buffer, format='JPEG')
    output_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
    
    return output_base64

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>CycleGAN Face-Sketch Converter</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .main-container {
                max-width: 1200px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.95);
                padding: 30px;
                border-radius: 20px;
                box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            }
            .title {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
                font-weight: 600;
            }
            .controls {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 15px;
                margin-bottom: 30px;
            }
            .image-container {
                background: white;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.05);
                margin: 10px;
                flex: 1;
            }
            .image-title {
                color: #34495e;
                margin-bottom: 15px;
                font-weight: 500;
            }
            #video {
                width: 100%;
                max-width: 500px;
                border-radius: 10px;
            }
            .btn-custom {
                background: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                transition: all 0.3s ease;
            }
            .btn-custom:hover {
                background: #2980b9;
                transform: translateY(-2px);
            }
            .btn-danger {
                background: #e74c3c;
            }
            .btn-danger:hover {
                background: #c0392b;
            }
            .file-upload {
                background: #fff;
                padding: 15px;
                border-radius: 10px;
                border: 2px dashed #3498db;
                text-align: center;
                margin-bottom: 20px;
            }
            .preview-image {
                max-width: 100%;
                height: auto;
                border-radius: 10px;
                margin-top: 10px;
            }
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            .loading-spinner {
                width: 40px;
                height: 40px;
            }
        </style>
    </head>
    <body>
        <div class="main-container">
            <h1 class="title">
                <i class="fas fa-exchange-alt"></i> 
                Face-Sketch Converter
            </h1>
            
            <div class="controls">
                <div class="row">
                    <div class="col-md-6">
                        <div class="file-upload">
                            <label for="fileInput" class="form-label">
                                <i class="fas fa-upload"></i> Choose an image
                            </label>
                            <input type="file" class="form-control" id="fileInput" accept="image/*">
                        </div>
                    </div>
                    <div class="col-md-6 text-center">
                        <button class="btn btn-custom me-2" onclick="startCamera()">
                            <i class="fas fa-camera"></i> Start Camera
                        </button>
                        <button class="btn btn-danger me-2" onclick="stopCamera()">
                            <i class="fas fa-stop"></i> Stop
                        </button>
                        <button class="btn btn-custom" onclick="captureImage()">
                            <i class="fas fa-camera-retro"></i> Capture
                        </button>
                    </div>
                </div>
            </div>

            <div class="loading">
                <div class="spinner-border text-primary loading-spinner" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p>Converting image...</p>
            </div>

            <div class="row">
                <div class="col-md-6">
                    <div class="image-container">
                        <h3 class="image-title">Input Image</h3>
                        <video id="video" autoplay style="display: none;"></video>
                        <canvas id="canvas" style="display: none;"></canvas>
                        <img id="inputImage" class="preview-image">
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="image-container">
                        <h3 class="image-title">Converted Image</h3>
                        <img id="outputImage" class="preview-image">
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            let video = document.getElementById('video');
            let canvas = document.getElementById('canvas');
            let context = canvas.getContext('2d');
            let stream = null;
            let loading = document.querySelector('.loading');

            // File input handling
            document.getElementById('fileInput').addEventListener('change', function(e) {
                if (e.target.files && e.target.files[0]) {
                    let reader = new FileReader();
                    reader.onload = function(event) {
                        document.getElementById('inputImage').src = event.target.result;
                        document.getElementById('video').style.display = 'none';
                        document.getElementById('inputImage').style.display = 'block';
                        loading.style.display = 'block';
                        
                        // Convert image to blob and send to server
                        fetch(event.target.result)
                            .then(res => res.blob())
                            .then(blob => {
                                let formData = new FormData();
                                formData.append('image', blob);
                                
                                fetch('/process', {
                                    method: 'POST',
                                    body: formData
                                })
                                .then(response => response.json())
                                .then(data => {
                                    loading.style.display = 'none';
                                    document.getElementById('outputImage').src = 'data:image/jpeg;base64,' + data.image;
                                })
                                .catch(error => {
                                    loading.style.display = 'none';
                                    alert('Error processing image');
                                    console.error('Error:', error);
                                });
                            });
                    };
                    reader.readAsDataURL(e.target.files[0]);
                }
            });

            // Camera handling with improved UI feedback
            async function startCamera() {
                try {
                    stream = await navigator.mediaDevices.getUserMedia({ video: true });
                    video.srcObject = stream;
                    video.style.display = 'block';
                    document.getElementById('inputImage').style.display = 'none';
                } catch (err) {
                    alert('Error accessing camera. Please make sure you have granted camera permissions.');
                    console.error("Error accessing camera:", err);
                }
            }

            function stopCamera() {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    video.style.display = 'none';
                    document.getElementById('inputImage').style.display = 'block';
                }
            }

            function captureImage() {
                if (stream) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    context.drawImage(video, 0, 0, canvas.width, canvas.height);
                    loading.style.display = 'block';
                    
                    canvas.toBlob(blob => {
                        let formData = new FormData();
                        formData.append('image', blob);
                        
                        fetch('/process', {
                            method: 'POST',
                            body: formData
                        })
                        .then(response => response.json())
                        .then(data => {
                            loading.style.display = 'none';
                            document.getElementById('outputImage').src = 'data:image/jpeg;base64,' + data.image;
                        })
                        .catch(error => {
                            loading.style.display = 'none';
                            alert('Error processing image');
                            console.error('Error:', error);
                        });
                    }, 'image/jpeg');
                }
            }
        </script>
    </body>
    </html>
    '''

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image_data = image_file.read()
    
    try:
        output_base64 = process_image(image_data)
        return jsonify({'image': output_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)