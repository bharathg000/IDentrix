from flask import Flask, render_template, Response, request, redirect, session, send_from_directory, abort
from ultralytics import YOLO
import cv2
import threading
import time
import csv
import os
import hmac
import secrets
import numpy as np
from flask import make_response

from compliance import check_compliance
from database import (
    init_database,
    get_all_people,
    get_today,
    get_filtered_violations,
    update_screenshot,
    delete_all_violations
)
from violation_engine import process_violation
from screenshot import save_violation_screenshot


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("COOKIE_SECURE", "false").lower() == "true",
    MAX_CONTENT_LENGTH=1024 * 1024,
)

if not os.getenv("FLASK_SECRET_KEY"):
    app.logger.warning("FLASK_SECRET_KEY is unset; sessions will not survive restarts.")

DATA_DIR = os.path.abspath(os.getenv(
    "IDENTRIX_DATA_DIR",
    os.path.dirname(os.path.abspath(__file__))
))
EVIDENCE_DIR = os.path.join(DATA_DIR, "violations")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# Initialize SQLite database
init_database()


# =====================================================
# MODELS
# =====================================================

id_model = YOLO("idcard_v2.pt")
person_model = YOLO("yolo11n.pt")


# =====================================================
# CAMERA
# =====================================================

def create_camera():
    """Open the first webcam source that produces a valid frame."""
    preferred_indexes = []

    env_index = os.getenv("IDENTRIX_CAMERA_INDEX")
    if env_index:
        try:
            preferred_indexes.append(int(env_index))
        except ValueError as exc:
            raise ValueError("IDENTRIX_CAMERA_INDEX must be an integer.") from exc

    preferred_indexes.extend([0, 1, 2, 3])
    backends = (
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("default", cv2.CAP_ANY),
    )

    seen = set()
    for index in preferred_indexes:
        if index in seen:
            continue
        seen.add(index)

        for backend_name, backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            success, frame = cap.read()
            if not success or frame is None or frame.size == 0:
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"Using camera index {index} with {backend_name}.")
            return cap

    return None


SERVER_CAMERA_ENABLED = os.getenv("IDENTRIX_SERVER_CAMERA", "false").lower() == "true"
camera = create_camera() if SERVER_CAMERA_ENABLED else None


# =====================================================
# SHARED DATA & STREAM STATE
# =====================================================

latest_frame = None
latest_id_boxes = []
latest_person_boxes = []
latest_id_card = False
frame_lock = threading.Lock()
latest_frame_time = 0

# Flag to track if the detect page / video stream is active
is_streaming = False
stream_lock = threading.Lock()
active_camera_token = None
active_camera_last_seen = 0


# =====================================================
# SETTINGS
# =====================================================

ID_DETECTION_INTERVAL = 0.5
ID_CONFIDENCE = 0.50

PERSON_DETECTION_INTERVAL = 1.0
PERSON_CONFIDENCE = 0.50

# Hold ID-card status so it doesn't blink
ID_HOLD_TIME = 1.5

# Violation processing interval
VIOLATION_CHECK_INTERVAL = 1.0


# =====================================================
# STATE
# =====================================================

last_id_seen_time = 0
last_violation_check = 0
logged_person_violations = set()


# =====================================================
# CAMERA THREAD
# =====================================================

def camera_thread():
    global latest_frame, latest_frame_time, active_camera_last_seen, camera

    while True:
        if not SERVER_CAMERA_ENABLED:
            time.sleep(1)
            continue

        if camera is None or not camera.isOpened():
            camera = create_camera()
            if camera is None:
                time.sleep(0.2)
                continue

        success, frame = camera.read()

        if success:
            with frame_lock:
                latest_frame = frame
                latest_frame_time = time.time()
            with stream_lock:
                if active_camera_token:
                    active_camera_last_seen = time.time()
        else:
            print("Camera read failed; retrying with a different capture source...")
            camera.release()
            camera = None
            time.sleep(0.2)


# =====================================================
# CARD -> PERSON MATCHING
# =====================================================

def card_belongs_to_person(card, person):
    card_center_x = (card["x1"] + card["x2"]) / 2
    card_center_y = (card["y1"] + card["y2"]) / 2

    return (
        person["x1"] <= card_center_x <= person["x2"]
        and person["y1"] <= card_center_y <= person["y2"]
    )


# =====================================================
# VIOLATION PROCESSING
# =====================================================

def process_current_violations():
    global logged_person_violations  # <--- Make sure this is declared here

    with frame_lock:
        if latest_frame is None:
            return

        frame = latest_frame.copy()
        people = [dict(person) for person in latest_person_boxes]

    current_person_ids = {p["id"] for p in people}
    
    # Clean up tracking for people who left the frame
    logged_person_violations = {pid for pid in logged_person_violations if pid in current_person_ids}

    for person in people:
        person_id = person["id"]
        has_id = person.get("has_id_card", False)

        if has_id:
            if person_id in logged_person_violations:
                logged_person_violations.remove(person_id)
            
            with frame_lock:
                for p in latest_person_boxes:
                    if p["id"] == person_id:
                        p["scan_started_at"] = time.time()
            continue

        elapsed = time.time() - person.get("scan_started_at", time.time())
        if elapsed < 5.0:
            continue

        if not has_id and person_id not in logged_person_violations:
            result = process_violation(person_id, "ID Card")

            if result["recorded"]:
                screenshot_path = save_violation_screenshot(
                    frame,
                    person,
                    person_id,
                    "ID Card"
                )

                if screenshot_path:
                    update_screenshot(result["violation_id"], screenshot_path)

                print(
                    f"NEW VIOLATION | Person {person_id} | "
                    f"ID Card violation recorded successfully."
                )

            logged_person_violations.add(person_id)
# =====================================================
# AI THREAD (Only runs when streaming is active)
# =====================================================

def ai_thread():
    global latest_id_boxes
    global latest_person_boxes
    global latest_id_card
    global last_id_seen_time
    global last_violation_check

    last_id_detection = 0
    last_person_detection = 0

    while True:
        with stream_lock:
            active = is_streaming

        with frame_lock:
            frame_is_fresh = (
                latest_frame is not None
                and time.time() - latest_frame_time <= 2
            )

        if not active or not frame_is_fresh:
            time.sleep(0.5)
            continue

        current_time = time.time()

        with frame_lock:
            if latest_frame is None:
                time.sleep(0.05)
                continue
            frame = latest_frame.copy()

        # PERSON DETECTION
        if current_time - last_person_detection >= PERSON_DETECTION_INTERVAL:
            last_person_detection = current_time

            try:
                results = person_model.track(
                    frame,
                    persist=True,
                    classes=[0],
                    conf=PERSON_CONFIDENCE,
                    imgsz=320,
                    device="cpu",
                    tracker="bytetrack_custom.yaml",
                    verbose=False
                )

                detected_people = []

                for box in results[0].boxes:
                    confidence = float(box.conf[0])
                    if confidence < PERSON_CONFIDENCE:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if box.id is None:
                        continue

                    person_id = int(box.id[0])

                    detected_people.append({
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "id": person_id,
                        "confidence": confidence,
                        "has_id_card": False,
                        "scan_started_at": time.time()
                    })

                with frame_lock:
                    previous_people = latest_person_boxes

                    for person in detected_people:
                        for old_person in previous_people:
                            if person["id"] == old_person["id"]:
                                person["has_id_card"] = old_person["has_id_card"]
                                person["scan_started_at"] = old_person.get(
                                    "scan_started_at", time.time()
                                )
                                break

                    latest_person_boxes = detected_people

            except Exception as e:
                print("Person model error:", e)

        # ID CARD DETECTION
        if current_time - last_id_detection >= ID_DETECTION_INTERVAL:
            last_id_detection = current_time

            try:
                results = id_model(
                    frame,
                    conf=ID_CONFIDENCE,
                    imgsz=320,
                    device="cpu",
                    verbose=False
                )

                detected_cards = []

                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if class_id == 0 and confidence >= ID_CONFIDENCE:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        detected_cards.append({
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "confidence": confidence,
                            "person_id": None
                        })

                with frame_lock:
                    people = latest_person_boxes
                    card_matched = False

                    for card in detected_cards:
                        for person in people:
                            if card_belongs_to_person(card, person):
                                card["person_id"] = person["id"]
                                person["has_id_card"] = True
                                card_matched = True
                                break

                    if card_matched:
                        last_id_seen_time = time.time()
                        latest_id_card = True
                        latest_id_boxes = detected_cards
                    else:
                        if time.time() - last_id_seen_time > ID_HOLD_TIME:
                            latest_id_boxes = []
                            latest_id_card = False

            except Exception as e:
                print("ID model error:", e)

        # VIOLATION CHECK
        if current_time - last_violation_check >= VIOLATION_CHECK_INTERVAL:
            last_violation_check = current_time

            try:
                process_current_violations()
            except Exception as e:
                print("Violation engine error:", e)

        time.sleep(0.01)


# =====================================================
# START THREADS
# =====================================================

camera_worker = None
if SERVER_CAMERA_ENABLED:
    camera_worker = threading.Thread(target=camera_thread, daemon=True)
    camera_worker.start()

ai_worker = threading.Thread(target=ai_thread, daemon=True)
ai_worker.start()


# =====================================================
# CORE ROUTES
# =====================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/healthz")
def health_check():
    return {"status": "ok"}


@app.route("/detect")
def detect():
    camera_token = session.setdefault("camera_token", secrets.token_urlsafe(32))
    return render_template(
        "detect.html",
        camera_token=camera_token,
        server_camera_enabled=SERVER_CAMERA_ENABLED
    )


def valid_camera_token():
    expected = session.get("camera_token", "")
    supplied = request.headers.get("X-Camera-Token", "")
    return bool(expected and supplied and hmac.compare_digest(expected, supplied))


@app.route("/api/camera/start", methods=["POST"])
def start_camera_session():
    global active_camera_token, active_camera_last_seen, is_streaming
    global latest_frame, latest_frame_time, latest_person_boxes, latest_id_boxes
    global latest_id_card, last_id_seen_time, logged_person_violations

    if not valid_camera_token():
        return {"error": "Refresh the page before starting the camera."}, 403

    token = session["camera_token"]
    now = time.time()
    with stream_lock:
        if (
            active_camera_token
            and active_camera_token != token
            and now - active_camera_last_seen < 15
        ):
            return {"error": "Another camera session is active. Try again shortly."}, 409

        if active_camera_token != token:
            with frame_lock:
                latest_person_boxes = []
                latest_id_boxes = []
                latest_id_card = False
                last_id_seen_time = 0
                logged_person_violations = set()
                if not SERVER_CAMERA_ENABLED:
                    latest_frame = None
                    latest_frame_time = 0

        active_camera_token = token
        active_camera_last_seen = now
        is_streaming = True

    return {"ok": True}


@app.route("/api/frame", methods=["POST"])
def receive_camera_frame():
    global latest_frame, latest_frame_time, active_camera_last_seen, is_streaming

    if not valid_camera_token():
        return {"error": "Camera session expired. Refresh the page and try again."}, 403

    token = session["camera_token"]
    with stream_lock:
        if active_camera_token != token:
            return {"error": "Start the camera before sending frames."}, 409

    if request.mimetype != "image/jpeg" or not request.data:
        return {"error": "Expected a non-empty JPEG frame."}, 400

    encoded = np.frombuffer(request.data, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "The uploaded camera frame could not be decoded."}, 400

    height, width = frame.shape[:2]
    if width > 1280 or height > 960:
        scale = min(1280 / width, 960 / height)
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

    with frame_lock:
        latest_frame = frame
        latest_frame_time = time.time()
    with stream_lock:
        active_camera_last_seen = time.time()
        is_streaming = True

    return {"ok": True}


@app.route("/api/camera/stop", methods=["POST"])
def stop_camera_session():
    global active_camera_token, active_camera_last_seen, is_streaming
    global latest_frame, latest_frame_time, latest_person_boxes, latest_id_boxes, latest_id_card

    if not valid_camera_token():
        return {"error": "Invalid camera session."}, 403

    with stream_lock:
        if active_camera_token == session["camera_token"]:
            active_camera_token = None
            active_camera_last_seen = 0
            is_streaming = False
            with frame_lock:
                latest_person_boxes = []
                latest_id_boxes = []
                latest_id_card = False
                if not SERVER_CAMERA_ENABLED:
                    latest_frame = None
                    latest_frame_time = 0

    return {"ok": True}


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        expected_username = os.getenv("ADMIN_USERNAME")
        expected_password = os.getenv("ADMIN_PASSWORD")

        if not expected_username or not expected_password:
            return render_template(
                "admin_login.html",
                error="Admin login is not configured. Set ADMIN_USERNAME and ADMIN_PASSWORD."
            ), 503

        if (
            hmac.compare_digest(username, expected_username)
            and hmac.compare_digest(password, expected_password)
        ):
            camera_token = session.get("camera_token")
            session.clear()
            if camera_token:
                session["camera_token"] = camera_token
            session["admin_logged_in"] = True
            return redirect("/admin/dashboard")

        return render_template(
            "admin_login.html",
            error="Invalid username or password"
        )

    return render_template("admin_login.html")


@app.route("/admin/dashboard")
def admin_dashboard():
    if not session.get("admin_logged_in"):
        return redirect("/admin/login")

    filter_date = request.args.get("filter_date", "").strip()
    sort_order = request.args.get("sort_order", "latest").strip()
    person_id_filter = request.args.get("person_id", "").strip()

    people = get_all_people()
    today_violations = get_filtered_violations(
        filter_date=filter_date,
        sort_order=sort_order,
        person_id=person_id_filter
    )

    return render_template(
        "admin_dashboard.html",
        people=people,
        today_violations=today_violations,
        filter_date=filter_date,
        sort_order=sort_order,
        person_id_filter=person_id_filter,
        csrf_token=session.setdefault("csrf_token", secrets.token_urlsafe(32))
    )
@app.route("/admin/history")
def admin_history():
    if not session.get("admin_logged_in"):
        return redirect("/admin/login")

    return redirect("/admin/dashboard#history")

@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect("/")


@app.route("/admin/violations/delete-all", methods=["POST"])
def delete_all_dashboard_violations():
    if not session.get("admin_logged_in"):
        return redirect("/admin/login")

    expected_token = session.get("csrf_token", "")
    supplied_token = request.form.get("csrf_token", "")
    if not expected_token or not hmac.compare_digest(expected_token, supplied_token):
        abort(400)

    screenshot_paths = delete_all_violations()
    evidence_roots = (
        os.path.realpath(EVIDENCE_DIR),
        os.path.realpath(os.path.join(app.static_folder, "violations")),
    )
    for screenshot_path in screenshot_paths:
        normalized_path = screenshot_path.replace("\\", "/")
        if normalized_path.startswith("static/violations/"):
            root = evidence_roots[1]
            relative_path = normalized_path[len("static/violations/"):]
        else:
            root = evidence_roots[0]
            if normalized_path.startswith("violations/"):
                normalized_path = normalized_path[len("violations/"):]
            relative_path = normalized_path

        full_path = os.path.realpath(os.path.join(root, relative_path))
        try:
            is_within_root = os.path.commonpath((root, full_path)) == root
        except ValueError:
            is_within_root = False
        if not is_within_root:
            app.logger.error("Skipped unsafe violation screenshot path: %s", screenshot_path)
            continue
        try:
            if os.path.isfile(full_path):
                os.remove(full_path)
        except OSError:
            app.logger.exception("Could not remove violation screenshot: %s", full_path)

    return redirect("/admin/dashboard")


@app.route("/admin/export/csv")
def export_csv():
    if not session.get("admin_logged_in"):
        return redirect("/admin/login")

    filter_date = request.args.get("filter_date", "").strip()
    sort_order = request.args.get("sort_order", "latest").strip()
    person_id_filter = request.args.get("person_id", "").strip()

    violations = get_filtered_violations(
        filter_date=filter_date,
        sort_order=sort_order,
        person_id=person_id_filter
    )

    output = []
    output.append(["Violation ID", "Person ID", "Violation Type", "Violation Date", "Count", "Last Detected", "Snapshot Path"])

    for v in violations:
        output.append([
            v["id"],
            v["person_id"],
            v["violation_type"],
            v["violation_date"],
            v["count"],
            v["last_detected"],
            v["screenshot_path"] or "None"
        ])

    response = make_response("")
    response.headers["Content-Disposition"] = "attachment; filename=compliance_audit_report.csv"
    response.headers["Content-Type"] = "text/csv"

    writer = csv.writer(response.stream)
    writer.writerows(output)

    return response


@app.route("/admin/screenshot/<path:filename>")
def violation_screenshot(filename):
    if not session.get("admin_logged_in"):
        abort(403)

    relative_path = filename.replace("\\", "/")
    if relative_path.startswith("static/violations/"):
        relative_path = relative_path[len("static/violations/"):]
        directory = os.path.join(app.static_folder, "violations")
    else:
        if relative_path.startswith("violations/"):
            relative_path = relative_path[len("violations/"):]
        directory = EVIDENCE_DIR
    return send_from_directory(directory, relative_path)


@app.route("/static/violations/<path:filename>")
def deny_public_violation_screenshot(filename):
    abort(404)


@app.route("/api/status")
def api_status():
    if (
        session.get("camera_token") != active_camera_token
        or time.time() - active_camera_last_seen > 2
    ):
        return {
            "result": "NO_PERSON",
            "message": "Start the camera to begin ID-card detection.",
            "badge_text": "WAITING",
            "badge_class": "rule-badge pending",
            "people": []
        }

    with frame_lock:
        people = [dict(p) for p in latest_person_boxes]

    if not people:
        return {
            "result": "NO_PERSON",
            "message": "No person detected in camera view.",
            "badge_text": "WAITING",
            "badge_class": "rule-badge pending",
            "people": []
        }

    person_statuses = []
    overall_status = "PASS"
    min_remaining = 5.0
    has_violation = False
    has_verifying = False

    for p in people:
        person_id = p["id"]
        has_id = p.get("has_id_card", False)
        elapsed = time.time() - p.get("scan_started_app", p.get("scan_started_at", time.time()))
        rem = max(0.0, 5.0 - elapsed)

        if has_id:
            p_status = "PASS"
            p_badge = "VERIFIED"
            p_class = "rule-badge ok"
        elif rem > 0:
            p_status = "VERIFYING"
            p_badge = f"VERIFYING ({rem:.1f}s)"
            p_class = "rule-badge pending"
            has_verifying = True
            if rem < min_remaining:
                min_remaining = rem
        else:
            p_status = "VIOLATION"
            p_badge = "VIOLATION"
            p_class = "rule-badge alert"
            has_violation = True

        person_statuses.append({
            "id": person_id,
            "has_id": has_id,
            "status": p_status,
            "badge_text": p_badge,
            "badge_class": p_class,
            "remaining": round(rem, 1)
        })

    # Set overall status based on independent evaluation
    if has_violation:
        status = "VIOLATION"
        message = "ID card missing after 5 seconds for one or more persons. Violation recorded."
        badge_text = "VIOLATION"
        badge_class = "rule-badge alert"
    elif has_verifying:
        status = "VERIFYING"
        message = f"Verifying ID cards. Violation will log in {min_remaining:.1f}s if missing."
        badge_text = f"VERIFYING ({min_remaining:.1f}s)"
        badge_class = "rule-badge pending"
    else:
        status = "PASS"
        message = "All detected students confirmed compliant."
        badge_text = "VERIFIED"
        badge_class = "rule-badge ok"

    return {
        "result": status,
        "message": message,
        "badge_text": badge_text,
        "badge_class": badge_class,
        "people": person_statuses
    }
# =====================================================
# VIDEO STREAM
# =====================================================

def generate_frames(camera_token):
    global is_streaming, latest_person_boxes, latest_id_boxes

    try:
        while True:
            with stream_lock:
                if active_camera_token != camera_token:
                    break

            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.01)
                    continue

                frame = latest_frame.copy()
                person_boxes = [dict(person) for person in latest_person_boxes]
                card_detected = latest_id_card

            for person in person_boxes:
                x1 = person["x1"]
                y1 = person["y1"]
                x2 = person["x2"]
                y2 = person["y2"]
                person_id = person["id"]
                has_id = person["has_id_card"]

                elapsed = time.time() - person.get("scan_started_at", time.time())
                remaining_time = max(0.0, 5.0 - elapsed)

                if has_id:
                    color = (105, 213, 160)
                    status_text = f"ID VERIFIED | Person #{person_id}"
                elif elapsed < 5.0:
                    color = (115, 224, 255)
                    status_text = f"VERIFYING ID ({remaining_time:.1f}s) | Person #{person_id}"
                else:
                    color = (123, 165, 255)
                    status_text = f"PENDING ID CARD | Person #{person_id}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
                
                label_size, _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, max(0, y1 - 28)), (x1 + label_size[0] + 16, y1), color, -1)
                cv2.putText(
                    frame,
                    status_text,
                    (x1 + 8, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (7, 11, 22),
                    1,
                    cv2.LINE_AA
                )

            success, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80]
            )

            if not success:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

            time.sleep(0.01)
    finally:
        with stream_lock:
            if not SERVER_CAMERA_ENABLED and time.time() - active_camera_last_seen >= 15:
                is_streaming = False


@app.route("/video")
def video():
    camera_token = session.get("camera_token")
    if not camera_token or camera_token != active_camera_token:
        abort(403)
    return Response(
        generate_frames(camera_token),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/result")
def result():
    if session.get("camera_token") != active_camera_token:
        abort(403)

    id_card = latest_id_card

    result_status, rules, issues = check_compliance(
        id_card=id_card
    )

    return render_template(
        "result.html",
        result=result_status,
        rules=rules,
        issues=issues
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )