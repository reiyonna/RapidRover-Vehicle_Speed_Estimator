from flask import Flask, render_template, request, redirect, url_for
import os
import subprocess
from werkzeug.utils import secure_filename
from main import ROOT_OUTPUT_DIR

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/videos' 

UPLOAD_PATH = os.path.join(os.path.dirname(__file__), UPLOAD_FOLDER)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), OUTPUT_FOLDER)

if not os.path.exists(UPLOAD_PATH):
    os.makedirs(UPLOAD_PATH)

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

app.config['UPLOAD_FOLDER'] = UPLOAD_PATH
app.config['OUTPUT_FOLDER'] = OUTPUT_PATH

def process_video(input_file):
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_file.filename)
    output_filename = 'new_generated_output_' + input_file.filename
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    input_file.save(input_path)
    input_file.save(output_path)
    # Run your video processing script here
    # Assuming your main.py is in the same directory
    command = f"python main.py -s {input_path}"
    subprocess.run(command, shell=True)
    return output_path


# def process_video(input_file):
#     # Create a unique folder for each video
#     video_folder = os.path.splitext(input_file.filename)[0]
#     output_folder = os.path.join(app.config['OUTPUT_FOLDER'], video_folder)

#     input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_file.filename)
#     output_filename = 'new_generated_output_' + input_file.filename
#     output_path = os.path.join(output_folder, output_filename)

#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     input_file.save(input_path)
#     input_file.save(output_path)

#     # Run your video processing script here
#     # Assuming your main.py is in the same directory
#     command = f"python main.py -s {input_path}"
#     subprocess.run(command, shell=True)

#     # Move the generated output file to the video folder
#     os.rename(output_path, os.path.join(output_folder, 'speed_result.txt'))

#     return os.path.join('/static/videos', video_folder, output_filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            output_path = process_video(file)
            return redirect(url_for('display_output', filename=file.filename))
    return render_template('index.html')  # Add this line to render the index.html template for GET requests

@app.route('/output/<filename>')
def display_output(filename):
    return render_template('test.html', filename=filename) #output.html - actual

@app.route('/hi')
def hi():
    return render_template('test.html')

if __name__ == '__main__':
    app.run(debug=True)

    