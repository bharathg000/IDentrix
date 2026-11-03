# IDentrix

**AI-Based Student ID Card Detection and Compliance System**

## Project Overview

IDentrix is an AI-powered computer vision application designed to detect student ID cards and monitor ID-card compliance in a campus environment.

The system uses object detection and a Flask web application to support ID-card detection, compliance monitoring, and administrative review.

## Key Features

- Student ID-card detection using YOLO.
- Real-time camera-based detection.
- ID-card compliance monitoring.
- Violation event recording and screenshot capture.
- Admin dashboard for reviewing detection records.

## Technologies Used

- Python
- Flask
- OpenCV
- YOLO
- PyTorch
- SQLite
- HTML and CSS

## Project Scope

IDentrix focuses exclusively on **student ID-card detection and compliance**. Hairstyle, clothing, footwear, and other appearance checks are outside the current project scope.

## Repository

GitHub: https://github.com/bharathg000/CampusAppearanceAI

## Run locally

Use Python 3.11, install the packages, and set admin credentials and a session secret:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:ADMIN_USERNAME = "your-admin-name"
$env:ADMIN_PASSWORD = "use-a-long-unique-password"
$env:FLASK_SECRET_KEY = "generate-a-long-random-secret"
python app.py
```

Open `http://localhost:5000` and press **Start camera** on the detection page. A browser camera requires permission; when hosted, browsers require HTTPS. Admin review is at `/admin/login`.

## Deploy with Docker

The app processes frames on the server CPU and keeps tracking state in memory, so deploy it as **one app instance with one Gunicorn worker**. Give the server enough memory for PyTorch and both YOLO models. The two required model files, `idcard_v2.pt` and `yolo11n.pt`, must be present in the repository/build context.

Build the image:

```sh
docker build -t identrix .
```

Start it with a persistent volume mounted at `/data`, replacing the example secrets:

```sh
docker volume create identrix-data
docker run -d --name identrix -p 8000:8000 --mount source=identrix-data,target=/data -e IDENTRIX_DATA_DIR=/data -e FLASK_SECRET_KEY='replace-with-a-long-random-secret' -e ADMIN_USERNAME='your-admin-name' -e ADMIN_PASSWORD='replace-with-a-strong-password' -e COOKIE_SECURE=true identrix
```

Put the app behind a hosting provider's HTTPS-enabled domain/reverse proxy. Do not expose port 8000 directly to the public internet without HTTPS. The persistent volume stores the SQLite database and private violation snapshots; back it up regularly. Do not run multiple replicas with SQLite or multiple Gunicorn workers because the live camera tracking state is process-local.

For local testing only, omit `COOKIE_SECURE=true` when accessing over plain HTTP. In production, set it to `true`. Camera frames are processed by the server and not retained as video; snapshots for recorded violations are stored in the persistent data directory and can only be viewed by an authenticated admin.
