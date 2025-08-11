from flask import Flask, flash, request, render_template, redirect, url_for
import sqlite3
from flask_socketio import SocketIO

from flask_sqlalchemy import SQLAlchemy
from flask import session
app = Flask(__name__)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user_registration.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy()
socketio = SocketIO(app)
db.init_app(app)  # Bind the database to the app


class UserRegistrationModel(db.Model):
    __tablename__ = 'user_registration'

    id = db.Column(db.Integer, primary_key=True)  # Primary Key
    name = db.Column(db.String(100), nullable=False)  # Name
    loginid = db.Column(db.String(100), nullable=False)  # Login ID (unique)
    password = db.Column(db.String(100), nullable=False)  # Password
    mobile = db.Column(db.String(10), nullable=False, unique=True)  # Mobile (unique)
    email = db.Column(db.String(100), nullable=False, unique=True)  # Email (unique)
    locality = db.Column(db.String(100), nullable=False)  # Locality
    state = db.Column(db.Text, nullable=False)  # Address
    status = db.Column(db.String(100), default='waiting')  # Status

    def __repr__(self):
        return f"<UserRegistrationModel {self.name}>"


# from flask import messages
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/logout")
def logout():
    session.clear()  # Clears the session, effectively logging the user out.
    return render_template("index.html")  # Renders the index.html template after logging out.


@app.route('/user_home')
def UserHome():
    return render_template('user/userhome.html')

@app.route('/user_register', methods=['GET', 'POST'])
def user_register():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        email = request.form['email']
        mobile = request.form['mobile']
        password = request.form['password']
        locality = request.form['locality']
        state = request.form['state']

        # Check if username or email already exists
        try:
            if UserRegistrationModel.query.filter_by(loginid=username).first():
                return render_template('register.html', message='Username already exists')
            if UserRegistrationModel.query.filter_by(email=email).first():
                return render_template('register.html', message='Email already exists')

            new_user = UserRegistrationModel(
                name=name,
                loginid=username,
                email=email,
                mobile=mobile,
                password=password,
                locality=locality,
                state=state
            )
            db.session.add(new_user)
            db.session.commit()

            return redirect(url_for('user_login'))
        except:
            return render_template('register.html')

    return render_template('register.html')

@app.route('/user_login', methods=['GET', 'POST'])
def user_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        try:
        
            user = UserRegistrationModel.query.filter_by(loginid=username).first()
            
            if not user:
                return render_template('login.html', message='Invalid details')
            if user.password != password:
                return render_template('login.html', message='Invalid details')
            if not user.status:
                return render_template('login.html', message='Status not activated')
                
            session['user_id'] = user.id
            return render_template('user/userhome.html')
        except:
            flash("error in valid details")
            return render_template('login.html')


    return render_template('login.html')

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username == 'admin' and password == 'admin':
            session['admin'] = True
            return redirect(url_for('admin_home'))
        return render_template('admin_login.html', message='Invalid credentials')
    return render_template('admin_login.html')

@app.route('/admin_home')
def admin_home():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    return render_template('adminhome.html')

@app.route('/user_details')
def user_details():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    users = UserRegistrationModel.query.all()
    return render_template('userdetails.html', users=users)

@app.route('/activate_user/<int:user_id>', methods=['POST'])
def activate_user(user_id):
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    user = UserRegistrationModel.query.get(user_id)
    if user:
        user.status = "activated"
        db.session.commit()
    return redirect(url_for('user_details'))


@app.route("/deactivate_user/<int:user_id>", methods=["POST"])
def deactivate_user(user_id):
    user = UserRegistrationModel.query.get_or_404(user_id)
    if user.status == "activated":  # Only deactivate if currently "activated"
        user.status = "waiting"
        db.session.commit()
        flash(f"User {user.name} has been deactivated successfully!", "info")
    return redirect("/user_details")



import os
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from werkzeug.utils import secure_filename


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
# Load the pre-trained model from the relative path in the media folder
model_path = os.path.join('media', 'my_model123.h5')
data = os.path.join('media','dataset')

# Load model without compilation
model = tf.keras.models.load_model(model_path, compile=False)

# Create optimizer with custom learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Compile the model with the custom optimizer
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)


    

@app.route('/training')
def training():
    # Adjust the path to the dataset directory inside the media folder
    dataset_dir = os.path.join(os.getcwd(), 'media', 'dataset1')   
    
    # Check if the dataset directory exists
    if not os.path.isdir(dataset_dir):
        return "Dataset directory does not exist. Please check the path."

    # Print directories and files once
    for root, dirs, files in os.walk(dataset_dir):
        for name in dirs:
            print(os.path.join(root, name))
        for name in files:
            print(os.path.join(root, name))
        break  
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    # Flow training images in batches of 32 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(227, 227),
        batch_size=32,
        class_mode='binary',
        shuffle=True  
    )
    # Build the CNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(227, 227, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Create optimizer with learning rate
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
    
    # Compile the model with the updated optimizer
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=50,
        verbose=1
    )
    
    loss = history.history['loss'][-1]
    accuracy = history.history['accuracy'][-1]
    
    print(f'Final loss: {loss}')
    print(f'Final accuracy: {accuracy}')
    return render_template('user/training.html', loss=loss, accuracy=accuracy)
    

import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


# Set the upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # Load the trained model (Ensure you provide the correct model path)
# model = load_model(r'D:\huggg\Deepcrack_A_Deep_Learning_Approach_for_Image-Based_Crack_\media\my_model123.h5')


@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('predict', filename=filename))
    return render_template('user/index1.html')

import cv2

def measure_crack(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding for better crack detection
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # Find the largest contour (assumed to be the crack)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit an ellipse to the contour to get better length/depth measurements
    if len(largest_contour) >= 5:  # Need at least 5 points to fit an ellipse
        ellipse = cv2.fitEllipse(largest_contour)
        # Get the major and minor axes of the ellipse
        (_, (major_axis, minor_axis), _) = ellipse
        
        # Major axis is the length, minor axis is the depth
        length = major_axis
        depth = minor_axis
    else:
        # Fallback to bounding rectangle if ellipse fitting is not possible
        rect = cv2.minAreaRect(largest_contour)
        (_, (width, height), _) = rect
        length = max(width, height)
        depth = min(width, height)
    
    # Convert pixels to millimeters (assuming a scale factor)
    scale_factor = 0.1  # millimeters per pixel
    length_mm = length * scale_factor
    depth_mm = depth * scale_factor
    
    return round(length_mm, 2), round(depth_mm, 2)

@app.route('/predict/<filename>')
def predict(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Check if file is an image or video
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(filepath)
        if img is None:
            return "Error: Unable to load image"
        
        # Resize image for model prediction to match training size (227x227)
        img_array = cv2.resize(img, (227, 227))
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)
        # Check if prediction[0][0] is greater than 0.5 for crack detection
        is_crack = prediction[0][0] > 0.5
        
        if is_crack:  # If crack is detected
            # Measure crack dimensions
            length, depth = measure_crack(img)
            if length and depth:
                return render_template(
                    'user/result1.html', 
                    filename=filename, 
                    detected=True,
                    length=length, 
                    depth=depth
                )

        else:
            return render_template(
                'user/result1.html',
                filename=filename,
                detected=False
            )
    
    else:
        message="Invalid format "
        return render_template('user/result1.html',messages=message)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True,port=8000)
