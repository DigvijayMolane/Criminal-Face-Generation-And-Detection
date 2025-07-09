from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, jsonify, Response
import os
import mysql.connector
from mysql.connector import Error
import torch
import cv2
import numpy as np
import clip
import face_recognition
from PIL import Image
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import json
from datetime import datetime, timedelta
import logging
import time
import base64
import re

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure secret key

# Load CLIP Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)

EMBEDDINGS_PATH = "D:/FaceRetrievalApp/image_embeddings.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = torch.load(EMBEDDINGS_PATH, map_location=device)
image_embeddings = torch.stack([torch.tensor(embedding).to(device) for embedding in data.values()])

# Folder configurations
OUTPUT_FOLDER = "static/generated_faces"
UPLOAD_FOLDER = 'static/images/'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv'}

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'rohit9146',
    'database': 'criminal_detection'
}

# Session configuration
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)  # Session lasts 24 hours
app.config['SESSION_COOKIE_SECURE'] = True  # For HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access to session cookie

def get_db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        print(f"Error connecting to MySQL Database: {e}")
        return None

def create_user(username, password):
    db = get_db_connection()
    if not db:
        return False
    
    cursor = db.cursor()
    try:
        hashed_password = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)",
            (username, hashed_password)
        )
        db.commit()
        return True
    except Error as e:
        print(f"Error creating user: {e}")
        return False
    finally:
        cursor.close()
        db.close()

def get_all_users():
    db = get_db_connection()
    if not db:
        return None
    
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, username, created_at FROM users")
        users = cursor.fetchall()
        return users
    except Error as e:
        print(f"Error fetching users: {e}")
        return None
    finally:
        cursor.close()
        db.close()

def verify_user_exists(username):
    db = get_db_connection()
    if not db:
        return False
    
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        return bool(user)
    except Error as e:
        print(f"Error verifying user: {e}")
        return False
    finally:
        cursor.close()
        db.close()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    # Clear any existing session when accessing the home page
    session.clear()
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        logger.debug(f"Login attempt for username: {username}")
        
        db = get_db_connection()
        if not db:
            flash("Database connection error")
            return render_template('login.html')
        
        cursor = db.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            
            logger.debug(f"User found in database: {user is not None}")
            
            if user:
                logger.debug("Checking password hash")
                logger.debug(f"Stored hash: {user['password']}")
                
                password_check = check_password_hash(user['password'], password)
                logger.debug(f"Password check result: {password_check}")
                
                if password_check:
                    session['username'] = username
                    flash('Logged in successfully!', 'success')
                    return redirect(url_for('dashboard'))
                else:
                    flash('Invalid password', 'error')
            else:
                flash('Invalid username', 'error')
            
            return render_template('login.html')
                
        except Exception as e:
            logger.error(f"Error during login: {str(e)}")
            flash(f"An error occurred: {str(e)}")
            return render_template('login.html')
        finally:
            cursor.close()
            db.close()
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    # Clear specific session data but keep some preferences if needed
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute('SELECT * FROM criminals')
    criminals = cursor.fetchall()
    cursor.close()
    db.close()
    return render_template('dashboard.html', criminals=criminals)

@app.route('/face_generation')
@login_required
def face_generation():
    return render_template('face_generation.html')

@app.route('/face_detection')
@login_required
def face_detection():
    return render_template('detect.html')

@app.route('/criminal_registration')
@login_required
def criminal_registration():
    return render_template('register.html')

@app.route('/register_form', methods=['GET'])
@login_required
def register_form():
    db = get_db_connection()
    if not db:
        flash("Database connection error")
        return redirect(url_for('login'))
    
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM criminals")
        criminals = cursor.fetchall()
        return render_template('register.html', criminals=criminals)
    except Error as e:
        flash(f"An error occurred: {str(e)}")
        return redirect(url_for('login'))
    finally:
        cursor.close()
        db.close()

@app.route('/register_criminal', methods=['POST'])
@login_required
def register_criminal():
    if 'username' not in session:
        flash("Please login to register criminals")
        return redirect(url_for('login'))

    db = get_db_connection()
    if not db:
        return jsonify({"error": "Database connection error"}), 500

    cursor = db.cursor()
    try:
        name = request.form['name']
        age = int(request.form['age'])
        gender = request.form['gender']
        crime = request.form['crime']
        image = request.files['image']

        if not image or not allowed_file(image.filename):
            return jsonify({"error": "Invalid file type"}), 400

        # Save image
        filename = secure_filename(f"{name}_{image.filename}")
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

        # Process face encoding
        img = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(img)
        
        if not face_locations:
            os.remove(image_path)
            flash("No face detected in the uploaded image")
            return redirect(url_for('register_form'))
            
        face_encodings = face_recognition.face_encodings(img, face_locations)
        if not face_encodings:
            os.remove(image_path)
            flash("Could not generate facial encoding")
            return redirect(url_for('register_form'))

        encoding_blob = face_encodings[0].tobytes()

        cursor.execute("""
            INSERT INTO criminals (name, age, gender, crime, image_path, facial_embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (name, age, gender, crime, image_path, encoding_blob))
        
        db.commit()
        flash("Criminal registered successfully")
        return redirect(url_for('register_form'))

    except Error as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    finally:
        cursor.close()
        db.close()

    
@app.route('/detect', methods=['GET', 'POST'])
@login_required
def detect():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded')
            return render_template('detect.html', detected=None)

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            flash('Invalid file type')
            return render_template('detect.html', detected=None)

        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            result = detect_faces(file_path)
            
            if result:
                alert_sound()
                return render_template('detect.html', 
                                     detected=True, 
                                     criminals=result['criminals'],
                                     detection_image=result['detected_image'])
            else:
                return render_template('detect.html', detected=False, uploaded_image=file_path)
        except Exception as e:
            flash(f"Error processing image: {str(e)}")
            return render_template('detect.html', detected=None)
    return render_template('detect.html', detected=None)

def detect_faces(file_path):
    try:
        # Load the image
        image = face_recognition.load_image_file(file_path)
        
        # Find all face locations in the image
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            print("No faces detected in the image")
            return None
            
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        # Convert image to OpenCV format
        image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        logger.debug(f"Processing image: {file_path}")
        logger.debug(f"Face locations found: {face_locations}")
        logger.debug(f"Number of faces detected: {len(face_encodings)}")
        
        # Get all criminals from database once
        db = get_db_connection()
        if not db:
            return None

        cursor = db.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM criminals")
            criminals = cursor.fetchall()
            
            best_match = None
            best_confidence = 0
            best_criminal = None
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                for criminal in criminals:
                    # Convert stored embedding from BLOB to numpy array
                    stored_encoding = np.frombuffer(criminal['facial_embedding'], dtype=np.float64)
                    
                    # Calculate face distance
                    face_distance = face_recognition.face_distance([stored_encoding], face_encoding)[0]
                    confidence = (1 - face_distance) * 100
                    
                    # Update best match if confidence is higher
                    if confidence > best_confidence and confidence > 60:  # Minimum 60% confidence threshold
                        best_confidence = confidence
                        best_match = (top, right, bottom, left)
                        best_criminal = criminal
            
            if best_match and best_criminal:
                # Draw rectangle around the best match only
                top, right, bottom, left = best_match
                cv2.rectangle(image_cv, (left, top), (right, bottom), (0, 0, 255), 2)
                
                # Add name label
                label = f"{best_criminal['name']} ({best_confidence:.1f}%)"
                cv2.putText(image_cv, label, (left, top - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                
                # Save the annotated image
                output_filename = f"detected_{os.path.basename(file_path)}"
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                cv2.imwrite(output_path, image_cv)
                
                return {
                    'criminals': [{
                        'name': best_criminal['name'],
                        'age': best_criminal['age'],
                        'crime_details': best_criminal['crime'],
                        'confidence': best_confidence,
                        'image_path': best_criminal['image_path']
                    }],
                    'detected_image': output_path
                }
                
        finally:
            cursor.close()
            db.close()
        
        return None

    except Exception as e:
        logger.error(f"Error in detect_faces: {str(e)}")
        return None

@app.route('/live')
@login_required
def live():
    return render_template('live.html')

@app.route('/video_feed')
@login_required
def video_feed():
    def generate():
        # Ensure uploads directory exists
        if not os.path.exists('static/uploads'):
            os.makedirs('static/uploads')

        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        try:
            while True:
                success, frame = camera.read()
                if not success:
                    break
                
                # Convert to RGB for face_recognition
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                print(f"Found {len(face_locations)} faces")  # Debug print
                
                # Draw rectangles and process faces
                for (top, right, bottom, left) in face_locations:
                    # Draw rectangle around face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Get face encoding
                    face_encoding = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])[0]
                    
                    # Get criminals from database
                    db = get_db_connection()
                    cursor = db.cursor(dictionary=True)
                    cursor.execute('SELECT * FROM criminals')
                    criminals = cursor.fetchall()
                    cursor.close()
                    db.close()
                    
                    # Check for matches
                    for criminal in criminals:
                        stored_encoding = np.frombuffer(criminal['facial_embedding'], dtype=np.float64)
                        face_distance = face_recognition.face_distance([stored_encoding], face_encoding)[0]
                        confidence = (1 - face_distance) * 100
                        
                        if confidence > 50:  # Adjust threshold as needed
                            # Draw red rectangle for criminal
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                            
                            # Add name and confidence
                            text = f"{criminal['name']} ({confidence:.1f}%)"
                            cv2.putText(frame, text, (left, top - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
                            # Save snapshot
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            snapshot_path = f"static/uploads/detection_{timestamp}.jpg"
                            cv2.imwrite(snapshot_path, frame)
                            
                            # Yield detection data
                            detection_data = {
                                'criminal': {
                                    'name': criminal['name'],
                                    'confidence': confidence,
                                    'snapshot': snapshot_path
                                }
                            }
                            yield f"data: {json.dumps(detection_data)}\n\n"
                
                # Convert frame to JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                # Yield frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
        except Exception as e:
            print(f"Error in video feed: {str(e)}")
        finally:
            camera.release()
            
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/detect_video', methods=['POST'])
@login_required
def detect_video():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file uploaded"}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "No video file selected"}), 400

        # Save video file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"temp_{timestamp}_{secure_filename(video_file.filename)}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)

        # Get frame position from request
        frame_position = request.form.get('frame_position', 0)
        frame_position = int(frame_position)

        # Process video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": "Failed to open video file"}), 500

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        ret, frame = cap.read()
        
        if not ret:
            cap.release()
            return jsonify({
                "status": "success",
                "criminal_detected": False,
                "message": "Video processing complete"
            })

        # Convert to RGB for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        # Draw boxes around all faces
        for (top, right, bottom, left) in face_locations:
            # Get face encoding
            face_encoding = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])[0]
            match = match_criminal(face_encoding)
            
            if match:
                # Draw red box for criminal
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                label = f"Criminal: {match['name']} ({match['confidence']:.1f}%)"
                cv2.putText(frame, label, (left, top - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Save snapshot with timestamp
                snapshot_filename = f"snapshot_{timestamp}.jpg"
                snapshot_path = os.path.join(app.config['UPLOAD_FOLDER'], snapshot_filename)
                cv2.imwrite(snapshot_path, frame)

                cap.release()
                return jsonify({
                    'status': 'success',
                    'criminal_detected': True,
                    'criminal_info': {
                        'name': match['name'],
                        'age': match['age'],
                        'crime': match['crime'],
                        'confidence': match['confidence']
                    },
                    'snapshot': f"static/images/{snapshot_filename}",  # Updated path
                    'timestamp': datetime.now().isoformat()
                })

        # Convert frame to base64 for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        progress = min(100, (frame_position / total_frames) * 100)

        cap.release()
        return jsonify({
            'status': 'processing',
            'frame': f'data:image/jpeg;base64,{frame_base64}',
            'progress': progress,
            'next_frame': frame_position + 1,
            'total_frames': total_frames
        })

    except Exception as e:
        print(f"Error in detect_video: {str(e)}")  # For debugging
        return jsonify({"error": str(e)}), 500


@app.route('/detect_image', methods=['POST'])
@login_required
def detect_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        result = detect_faces(file_path)
        
        if result:
            return jsonify({
                'detected': True,
                'criminals': result['criminals'],
                'detection_image': result['detected_image']
            })
        else:
            return jsonify({
                'detected': False,
                'message': 'No criminals detected',
                'uploaded_image': file_path
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/edit_criminal/<int:id>', methods=['GET', 'POST'])
def edit_criminal(id):
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    db = get_db_connection()
    if not db:
        return jsonify({"error": "Database connection error"}), 500

    cursor = db.cursor(dictionary=True)
    
    if request.method == 'GET':
        try:
            cursor.execute("""
                SELECT id, name, age, gender, crime, image_path 
                FROM criminals 
                WHERE id = %s
            """, (id,))
            
            criminal = cursor.fetchone()
            if criminal:
                return jsonify({
                    'id': criminal['id'],
                    'name': criminal['name'],
                    'age': criminal['age'],
                    'gender': criminal['gender'],
                    'crime': criminal['crime'],
                    'image_path': criminal['image_path']
                })
            return jsonify({"error": "Criminal not found"}), 404
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            cursor.close()
            db.close()
    
    elif request.method == 'POST':
        try:
            name = request.form['name']
            age = int(request.form['age'])
            gender = request.form['gender']
            crime = request.form['crime']
            
            # Start with basic update query
            update_query = """
                UPDATE criminals 
                SET name = %s, age = %s, gender = %s, crime = %s
                WHERE id = %s
            """
            update_values = (name, age, gender, crime, id)
            
            # Handle image update if new image is provided
            if 'image' in request.files and request.files['image'].filename:
                image = request.files['image']
                if allowed_file(image.filename):
                    # Get old image path
                    cursor.execute("SELECT image_path FROM criminals WHERE id = %s", (id,))
                    old_image = cursor.fetchone()
                    
                    # Delete old image if it exists
                    if old_image and old_image['image_path']:
                        try:
                            old_image_path = old_image['image_path']
                            if os.path.exists(old_image_path):
                                os.remove(old_image_path)
                        except Exception as e:
                            print(f"Error deleting old image: {e}")

                    # Save new image
                    filename = secure_filename(f"{name}_{image.filename}")
                    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    image.save(image_path)
                    
                    # Update query to include new image path
                    update_query = """
                        UPDATE criminals 
                        SET name = %s, age = %s, gender = %s, crime = %s, image_path = %s
                        WHERE id = %s
                    """
                    update_values = (name, age, gender, crime, image_path, id)

            # Execute the update query
            cursor.execute(update_query, update_values)
            db.commit()
            
            flash("Criminal record updated successfully")
            return redirect(url_for('register_form'))
            
        except Exception as e:
            db.rollback()
            flash(f"Error updating record: {str(e)}")
            return redirect(url_for('register_form'))
        finally:
            cursor.close()
            db.close()

    return jsonify({"error": "Invalid request method"}), 405

@app.route('/delete_criminal/<int:id>', methods=['POST'])
def delete_criminal(id):
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    db = get_db_connection()
    if not db:
        return jsonify({"error": "Database connection error"}), 500

    cursor = db.cursor()
    try:
        # Get image path before deletion
        cursor.execute("SELECT image_path FROM criminals WHERE id = %s", (id,))
        result = cursor.fetchone()
        if result and result[0]:
            # Delete the image file
            image_path = result[0]
            if os.path.exists(image_path):
                os.remove(image_path)

        # Delete the record
        cursor.execute("DELETE FROM criminals WHERE id = %s", (id,))
        db.commit()
        return jsonify({"message": "Criminal deleted successfully"})
    except Error as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    finally:
        cursor.close()
        db.close()

@app.route('/check_criminal/<int:id>', methods=['GET'])
def check_criminal(id):
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    db = get_db_connection()
    if not db:
        return jsonify({"error": "Database connection error"}), 500

    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, name, age, gender, crime, image_path FROM criminals WHERE id = %s", (id,))
        criminal = cursor.fetchone()
        if criminal:
            # Convert to serializable format
            return jsonify({
                'id': criminal['id'],
                'name': criminal['name'],
                'age': criminal['age'],
                'gender': criminal['gender'],
                'crime': criminal['crime'],
                'image_path': criminal['image_path']
            })
        return jsonify({"error": "Criminal not found"}), 404
    except Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        db.close()

@app.route('/debug_db')
def debug_db():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    db = get_db_connection()
    if not db:
        return jsonify({"error": "Database connection error"}), 500

    cursor = db.cursor(dictionary=True)
    try:
        # Check table structure
        cursor.execute("DESCRIBE criminals")
        structure = cursor.fetchall()
        
        # Get sample data
        cursor.execute("SELECT * FROM criminals LIMIT 1")
        sample = cursor.fetchone()
        
        return jsonify({
            "table_structure": structure,
            "sample_data": sample
        })
    except Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        db.close()

@app.route('/test_db')
def test_db():
    try:
        db = get_db_connection()
        if db:
            cursor = db.cursor(dictionary=True)
            cursor.execute("SELECT COUNT(*) as count FROM criminals")
            result = cursor.fetchone()
            cursor.close()
            db.close()
            return jsonify({
                "status": "success",
                "message": "Database connection successful",
                "criminal_count": result['count']
            })
        return jsonify({
            "status": "error",
            "message": "Could not connect to database"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

def verify_dataset():
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"Embeddings file not found at {EMBEDDINGS_PATH}")
    
    if not os.path.exists("dataset/images"):
        raise FileNotFoundError("Dataset images folder not found")
    
    image_files = os.listdir("dataset/images")
    if not image_files:
        raise ValueError("No images found in dataset folder")
    
    print(f"Dataset verified: {len(image_files)} images found")

# Call this when your app starts
try:
    verify_dataset()
except Exception as e:
    print(f"Dataset verification failed: {str(e)}")

@app.route('/test_camera')
def test_camera():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({"error": "Failed to open camera"}), 500
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({"error": "Failed to read from camera"}), 500
            
        return jsonify({"success": "Camera is working"}), 200
    except Exception as e:
        return jsonify({"error": f"Camera error: {str(e)}"}), 500

@app.route('/setup_db')
def setup_db():
    try:
        db = get_db_connection()
        if not db:
            return "Failed to connect to database"
        
        cursor = db.cursor()
        
        # Create users table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create criminals table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS criminals (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                age INT NOT NULL,
                gender VARCHAR(50) NOT NULL,
                crime TEXT NOT NULL,
                image_path VARCHAR(255) NOT NULL,
                facial_embedding MEDIUMBLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create a test user if no users exist
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        if user_count == 0:
            # Create a default user (username: admin, password: admin123)
            hashed_password = generate_password_hash('admin123')
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (%s, %s)",
                ('admin', hashed_password)
            )
        
        db.commit()
        
        # Check tables
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        # Get user count
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        # Get criminal count
        cursor.execute("SELECT COUNT(*) FROM criminals")
        criminal_count = cursor.fetchone()[0]
        
        cursor.close()
        db.close()
        
        return jsonify({
            "status": "success",
            "tables": [table[0] for table in tables],
            "user_count": user_count,
            "criminal_count": criminal_count,
            "message": "Database setup complete"
        })
        
    except Error as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# Add a route to check existing users
@app.route('/check_users')
def check_users():
    try:
        db = get_db_connection()
        if not db:
            return "Failed to connect to database"
        
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT id, username, created_at FROM users")
        users = cursor.fetchall()
        
        cursor.close()
        db.close()
        
        return jsonify({
            "status": "success",
            "users": users
        })
        
    except Error as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# Add a route to create a new user
@app.route('/create_test_user')
def create_test_user():
    try:
        db = get_db_connection()
        if not db:
            return "Failed to connect to database"
        
        cursor = db.cursor()
        
        # Create a test user
        username = 'admin'
        password = 'admin123'
        hashed_password = generate_password_hash(password)
        
        try:
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (%s, %s)",
                (username, hashed_password)
            )
            db.commit()
            message = "Test user created successfully"
        except Error as e:
            if e.errno == 1062:  # Duplicate entry error
                message = "Test user already exists"
            else:
                raise e
        
        cursor.close()
        db.close()
        
        return jsonify({
            "status": "success",
            "message": message,
            "test_credentials": {
                "username": username,
                "password": password
            }
        })
        
    except Error as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/create_user/<username>/<password>')
def create_user_route(username, password):
    try:
        if create_user(username, password):
            return jsonify({
                "status": "success",
                "message": f"User {username} created successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to create user"
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/reset_admin')
def reset_admin():
    try:
        db = get_db_connection()
        if not db:
            return jsonify({"error": "Database connection failed"})
        
        cursor = db.cursor()
        try:
            # Delete existing admin user
            cursor.execute("DELETE FROM users WHERE username = 'admin'")
            
            # Create new admin user
            password = 'admin123'
            hashed_password = generate_password_hash(password)
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (%s, %s)",
                ('admin', hashed_password)
            )
            
            db.commit()
            
            # Verify the user was created
            cursor.execute("SELECT username, password FROM users WHERE username = 'admin'")
            user = cursor.fetchone()
            
            return jsonify({
                "status": "success",
                "message": "Admin user reset successfully",
                "credentials": {
                    "username": "admin",
                    "password": "admin123"
                },
                "verification": {
                    "user_exists": user is not None,
                    "password_hash": user[1] if user else None
                }
            })
            
        finally:
            cursor.close()
            db.close()
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/video_progress')
def video_progress():
    def generate():
        while True:
            # Get progress from a global variable or cache
            progress = 0  # You'll need to implement progress tracking
            yield f"data: {json.dumps({'progress': progress})}\n\n"
            if progress >= 100:
                break
            time.sleep(1)
    return Response(generate(), mimetype='text/event-stream')

# Add this function to create a default alert sound
def create_default_alert_sound():
    try:
        alert_path = os.path.join(app.static_folder, 'alert.wav')
        if not os.path.exists(alert_path):
            import numpy as np
            from scipy.io import wavfile
            
            # Generate a simple beep sound
            sample_rate = 44100
            duration = 0.5  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration))
            # Generate a 440 Hz sine wave
            frequency = 440
            beep = np.sin(2 * np.pi * frequency * t)
            # Add a second frequency for more complex sound
            beep += np.sin(2 * np.pi * 880 * t) * 0.5
            # Normalize and convert to 16-bit integer
            beep = np.int16(beep * 32767)
            # Save the file
            wavfile.write(alert_path, sample_rate, beep)
            logger.info(f"Created default alert sound at {alert_path}")
    except Exception as e:
        logger.error(f"Error creating alert sound: {e}")

@app.route('/process_frame', methods=['POST'])
@login_required
def process_frame():
    try:
        print("Processing new frame...")  # Debug log
        
        # Get frame data
        data = request.json
        image_data = data['frame'].split(',')[1]
        
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("Failed to decode frame")
            return jsonify({'error': 'Invalid frame data'}), 400
            
        print("Frame shape:", frame.shape)  # Debug log
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        print(f"Detected {len(face_locations)} faces")  # Debug log
        
        results = []
        if face_locations:
            # Get encodings
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            print(f"Got {len(face_encodings)} face encodings")  # Debug log
            
            # Get criminals from database
            db = get_db_connection()
            cursor = db.cursor(dictionary=True)
            cursor.execute('SELECT * FROM criminals')
            criminals = cursor.fetchall()
            cursor.close()
            db.close()
            
            print(f"Found {len(criminals)} criminals in database")  # Debug log
            
            for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                top, right, bottom, left = face_location
                
                # Check each criminal
                for criminal in criminals:
                    try:
                        stored_encoding = np.frombuffer(criminal['facial_embedding'], dtype=np.float64)
                        distance = face_recognition.face_distance([stored_encoding], face_encoding)[0]
                        confidence = (1 - distance) * 100
                        
                        print(f"Comparing with {criminal['name']}, confidence: {confidence:.2f}%")  # Debug log
                        
                        if confidence > 60:  # Adjusted threshold
                            results.append({
                                'location': [top, right, bottom, left],
                                'criminal': {
                                    'name': criminal['name'],
                                    'age': criminal['age'],
                                    'crime': criminal['crime'],
                                    'confidence': confidence
                                }
                            })
                            print(f"Match found: {criminal['name']}")  # Debug log
                            break
                    except Exception as e:
                        print(f"Error comparing with criminal: {str(e)}")
                        continue
                
                if not any(r.get('criminal') for r in results):
                    results.append({
                        'location': [top, right, bottom, left],
                        'criminal': None
                    })
        
        print(f"Returning {len(results)} results")  # Debug log
        return jsonify({'results': results})
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/check_criminals')
@login_required
def check_criminals():
    try:
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        cursor.execute('SELECT id, name FROM criminals')
        criminals = cursor.fetchall()
        cursor.close()
        db.close()
        
        print(f"Found {len(criminals)} criminals in database")  # Debug print
        return jsonify({
            'count': len(criminals),
            'criminals': [{'id': c['id'], 'name': c['name']} for c in criminals]
        })
    except Exception as e:
        print(f"Error checking criminals: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/retrieve', methods=['GET', 'POST'])
def retrieve():
    image_url = None
    if request.method == 'POST':
        text_input = request.form.get('description')
        if text_input:
            try:
                text_token = clip.tokenize([text_input]).to(device)
                with torch.no_grad():
                    text_embedding = model.encode_text(text_token)

                similarities = (image_embeddings @ text_embedding.T).squeeze(1)
                best_match_idx = similarities.argmax().item()

                image_files = os.listdir("dataset/images/")
                best_match_image = image_files[best_match_idx]

                image_path = os.path.join("dataset/images", best_match_image)
                output_path = os.path.join(OUTPUT_FOLDER, best_match_image)
                Image.open(image_path).save(output_path)

                image_url = url_for('static', filename=f'generated_faces/{best_match_image}')
            except Exception as e:
                flash(f"Error processing image: {str(e)}")
    
    return render_template('retrieve.html', image_url=image_url)

def match_criminal(face_encoding):
    # Get all criminals from database
    db = get_db_connection()
    if not db:
        return None
        
    try:
        cursor = db.cursor(dictionary=True)
        cursor.execute('SELECT * FROM criminals')
        criminals = cursor.fetchall()
        
        best_match = None
        best_distance = float('inf')
        
        for criminal in criminals:
            stored_encoding = np.frombuffer(criminal['facial_embedding'], dtype=np.float64)
            face_distance = face_recognition.face_distance([stored_encoding], face_encoding)[0]
            
            if face_distance < best_distance:
                best_distance = face_distance
                best_match = criminal

        if best_distance < 0.5:
            return {
                'name': best_match['name'],
                'age': best_match['age'],
                'crime': best_match['crime'],
                'confidence': ((1 - best_distance) * 100)
            }
        return None
            
    except Exception as e:
        logger.error(f"Error in match_criminal: {str(e)}")
        return None
    finally:
        cursor.close()
        db.close()

# Add this function before the detect route
def alert_sound():
    try:
        # Create alert sound file if it doesn't exist
        create_default_alert_sound()
        return jsonify({"status": "success", "message": "Alert sound played"})
    except Exception as e:
        logger.error(f"Error playing alert sound: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/test_password/<username>/<password>')
def test_password(username, password):
    try:
        db = get_db_connection()
        if not db:
            return jsonify({"error": "Database connection failed"})
        
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        
        if not user:
            return jsonify({
                "status": "error",
                "message": "User not found"
            })
            
        password_check = check_password_hash(user['password'], password)
        
        return jsonify({
            "status": "success",
            "username": username,
            "stored_hash": user['password'],
            "password_check": password_check
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    finally:
        cursor.close()
        db.close()

# This should be at the end of the file
if __name__ == '__main__':
    create_default_alert_sound()
    app.run(debug=True)
