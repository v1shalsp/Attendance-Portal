# 👁️‍🗨️ Face Recognition Attendance Portal  
### *Automated Attendance System using Deep Learning and Cloud Deployment*  

---

## 🧠 Overview

The **Face Recognition Attendance Portal** is an AI-powered system that automates attendance marking using **facial recognition**.  
It recognizes individuals in real time through live video or uploaded images and logs attendance automatically into a **cloud database**.  

The system leverages **FaceNet** for facial recognition and **MTCNN** for detection, offering a secure, scalable, and efficient alternative to manual attendance tracking.

---

## 🎯 Objectives

- Automate attendance marking using face recognition.  
- Reduce human effort and eliminate proxy attendance.  
- Provide web/mobile access to attendance records.  
- Deploy securely on cloud infrastructure for scalability.

---

## ⚙️ Tech Stack

| Layer | Technologies Used |
|-------|--------------------|
| **Frontend** | React / Flutter |
| **Backend** | FastAPI (REST APIs) |
| **Database** | PostgreSQL |
| **Cloud Storage** | AWS S3 |
| **Model** | FaceNet (Recognition), MTCNN (Detection) |
| **Language** | Python |
| **Libraries** | TensorFlow, OpenCV, NumPy, SQLAlchemy |
| **Deployment** | AWS EC2 + S3 |

---

## 🧠 Model Details

### 🔹 1. Face Detection – MTCNN
- Detects and crops faces from video frames.  
- Handles multiple faces and varied lighting.  
- Provides bounding boxes and keypoints.

### 🔹 2. Face Recognition – FaceNet
- Generates **128-dimensional embeddings** per face.  
- Compares embeddings using **cosine similarity** to identify users.  
- Uses a similarity threshold (≥ 0.7 → valid match).

---

## 🧾 Backend Endpoints (FastAPI)

| Endpoint | Method | Description |
|-----------|--------|-------------|
| `/register_user` | POST | Stores new user face embedding and image |
| `/mark_attendance` | POST | Recognizes face and marks attendance |
| `/get_attendance_logs` | GET | Fetches attendance history |
| `/update_profile` | PUT | Updates user information |

---

## 🗄️ Database Schema (PostgreSQL)

| Table | Fields |
|--------|--------|
| `users` | `user_id`, `name`, `email`, `embedding_vector`, `face_image_path` |
| `attendance_logs` | `log_id`, `user_id`, `timestamp`, `camera_source` |

---

## 📊 Features

✅ Real-time facial recognition through webcam or app  
✅ REST-based backend APIs  
✅ Cloud storage for images and embeddings  
✅ Secure and centralized attendance tracking  
✅ Web/mobile dashboard for users and admins  
✅ Multi-user and multi-camera support  

---

## ⚡ Workflow

1️⃣ **User Registration** → Capture face → Generate embedding → Store in DB & S3  
2️⃣ **Attendance Session** → Detect & recognize face → Verify similarity  
3️⃣ **Logging** → Record timestamp and source camera  
4️⃣ **Dashboard** → Display attendance analytics for users/admins  

---

## 📈 Results & Impact

| Metric | Value |
|---------|--------|
| Recognition Accuracy | **~93%** |
| False Positive Rate | < 5% |
| Latency per face | ~0.8s |
| Efficiency Improvement | **70% reduction** in manual effort |

✅ **Deployed successfully** on AWS EC2 with PostgreSQL and S3 integration.
