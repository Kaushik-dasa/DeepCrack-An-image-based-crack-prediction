"# DeepCrack: AI-Powered Crack Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🔍 Overview

DeepCrack is an advanced AI-powered web application that uses deep learning to detect and analyze cracks in structural images. The system employs a Convolutional Neural Network (CNN) to identify cracks and provides detailed measurements including length and depth analysis.

## ✨ Features

### 🎯 Core Functionality
- **AI-Powered Detection**: Advanced CNN model for accurate crack identification
- **Crack Measurement**: Automatic calculation of crack length and depth in millimeters
- **Real-time Analysis**: Instant prediction results with visual feedback
- **Multi-format Support**: Supports PNG, JPG, and JPEG image formats

### 👥 User Management
- **User Registration & Authentication**: Secure user account system
- **Admin Panel**: Administrative controls for user management
- **Session Management**: Secure login/logout functionality
- **User Dashboard**: Personalized user experience with prediction history

### 🎨 Interactive Interface
- **Modern UI**: Responsive design with dynamic visual effects
- **Real-time Animations**: Interactive crack effects on mouse movement
- **Mobile Friendly**: Optimized for various screen sizes
- **Intuitive Navigation**: Easy-to-use interface for all user levels

## 🏗️ Architecture

### Model Architecture
```
Input Layer (227x227x3)
    ↓
Conv2D (32 filters, 3x3) + ReLU + MaxPooling2D
    ↓
Conv2D (64 filters, 3x3) + ReLU + MaxPooling2D
    ↓
Conv2D (128 filters, 3x3) + ReLU + MaxPooling2D
    ↓
Flatten + Dense (128) + ReLU
    ↓
Dense (1) + Sigmoid (Binary Classification)
```

### Technology Stack
- **Backend**: Flask (Python web framework)
- **AI/ML**: TensorFlow/Keras for deep learning
- **Database**: SQLite with SQLAlchemy ORM
- **Frontend**: HTML5, CSS3, JavaScript
- **Image Processing**: OpenCV for crack measurement
- **File Handling**: Werkzeug for secure file uploads

## 📊 Dataset

The application includes two comprehensive datasets:

### Dataset 1 (`media/dataset/`)
- **Positive Samples**: 154 crack images
- **Negative Samples**: 100 non-crack images
- **Purpose**: Initial training and validation

### Dataset 2 (`media/dataset1/`)
- **Positive Samples**: 20,000 crack images
- **Negative Samples**: 20,000 non-crack images
- **Purpose**: Large-scale training for improved accuracy

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)
- 4GB+ RAM recommended
- GPU support optional but recommended for training

### Step 1: Clone the Repository
```bash
git clone https://github.com/Kaushik-dasa/DeepCrack-An-image-based-crack-prediction.git
cd DeepCrack-An-image-based-crack-prediction
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: The requirements.txt contains 245+ packages. Key dependencies include:
- Flask 3.0.3
- TensorFlow 2.10.0
- OpenCV 4.9.0.80
- SQLAlchemy 2.0.30
- NumPy 1.24.4
- Pillow 10.3.0

### Step 4: Initialize Database
```bash
python create_table.py
```

### Step 5: Run the Application
```bash
python app.py
```

The application will be available at `http://localhost:8000`

## 📱 Usage

### For Users

1. **Registration**: Create a new account via the registration page
2. **Login**: Access your account using your credentials
3. **Upload Image**: Navigate to the prediction page and upload an image
4. **View Results**: Get instant crack detection results with measurements
5. **History**: View your prediction history in the user dashboard

### For Administrators

1. **Admin Login**: Access the admin panel with admin credentials
2. **User Management**: View and manage user accounts
3. **System Monitoring**: Monitor application usage and performance

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/user_register` | GET/POST | User registration |
| `/user_login` | GET/POST | User login |
| `/admin_login` | GET/POST | Admin login |
| `/upload_file` | GET/POST | Image upload and prediction |
| `/predict/<filename>` | GET | Prediction results |
| `/training` | GET | Model training interface |

## 🧠 Model Details

### Training Configuration
- **Input Size**: 227×227×3 pixels
- **Batch Size**: 32
- **Epochs**: 50
- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Binary Crossentropy
- **Activation**: ReLU (hidden layers), Sigmoid (output)

### Crack Measurement Algorithm
The system uses advanced computer vision techniques:

1. **Preprocessing**: Convert to grayscale and apply Gaussian blur
2. **Thresholding**: Adaptive thresholding for crack detection
3. **Contour Detection**: Find crack boundaries
4. **Ellipse Fitting**: Calculate crack dimensions
5. **Measurement**: Convert pixels to millimeters using scale factor

## 📁 Project Structure

```
DeepCrack/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── HOW TO RUN            # Quick start guide
├── database.db           # SQLite database
├── media/
│   ├── my_model123.h5    # Trained model (Git LFS)
│   ├── dataset/          # Training dataset 1
│   └── dataset1/         # Training dataset 2
├── static/
│   ├── uploads/          # User uploaded images
│   └── 2.webp           # Static assets
├── templates/
│   ├── index.html        # Home page
│   ├── login.html        # User login
│   ├── register.html     # User registration
│   ├── admin_login.html  # Admin login
│   └── user/
│       ├── userhome.html # User dashboard
│       ├── index1.html   # Upload interface
│       ├── result1.html  # Results page
│       └── training.html # Training interface
└── instance/
    └── user_registration.db # User database
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file for production deployment:
```env
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///production.db
UPLOAD_FOLDER=static/uploads
MAX_CONTENT_LENGTH=16777216  # 16MB max file size
```

### Model Configuration
The pre-trained model is stored using Git LFS due to its size (169MB). Ensure Git LFS is installed:
```bash
git lfs install
git lfs pull
```

## 🚀 Deployment

### Local Development
```bash
python app.py
# Access at http://localhost:8000
```

### Production Deployment

#### Using Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

#### Using Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "app.py"]
```

## 🧪 Testing

### Manual Testing
1. Upload various image formats (PNG, JPG, JPEG)
2. Test with crack and non-crack images
3. Verify measurement accuracy
4. Test user registration and login flows

### Automated Testing
```bash
# Install testing dependencies
pip install pytest pytest-flask

# Run tests
pytest tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Kaushik Dasa**
- GitHub: [@Kaushik-dasa](https://github.com/Kaushik-dasa)
- Repository: [DeepCrack-An-image-based-crack-prediction](https://github.com/Kaushik-dasa/DeepCrack-An-image-based-crack-prediction)

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Flask community for the web framework
- OpenCV contributors for computer vision tools
- The research community for crack detection methodologies

## 📞 Support

For support, please open an issue on GitHub or contact the maintainer.

---

**⭐ If you find this project helpful, please give it a star on GitHub!**"
