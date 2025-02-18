# 🏆 FaceEngine  

FaceEngine is a powerful face recognition and tracking system using **DeepFace, DeepSort, and YoloFace**. This guide will help you set up the project quickly.  

---

## 📌 Installation Guide  

### **1️⃣ Install All Dependencies**  
Run the following script to install required packages:  
```sh
./install_repo.sh
```
Then, activate the virtual environment:  
```sh
source env/bin/activate
```

---

### **2️⃣ Install DeepFace (Face Recognition)**  
DeepFace is used for **face recognition and analysis**. Install it using:  
```sh
pip install deepface
```
Alternatively, install the latest version from the source code:  
```sh
git clone https://github.com/serengil/deepface.git
cd deepface
pip install -e .
```

---

### **3️⃣ Install DeepSort (For Tracking)**  
DeepSort is used for **tracking faces** after detection. Install it by running:  
```sh
git clone https://github.com/nwojke/deep_sort.git
```

---

### **4️⃣ Install YoloFace (Face Detection)**  
YoloFace is used for **detecting faces in images and videos**. Install it as follows:  
```sh
git clone https://github.com/elyha7/yoloface.git
cd yoloface
pip install -r requirements.txt
```

---

