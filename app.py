from flask import Flask, render_template, Response, request, redirect, session
from ultralytics import YOLO
import cv2
import threading
import time
import csv
from flask import make_response

from compliance import check_compliance
from database import (
    init_database,
    get_all_people,
    get_today,
    get_filtered_violations,
    update_screenshot
)
from violation_engine import process_violation
from screenshot import save_violation_screenshot


app = Flask(__name__)
app.secret_key = "campusappearanceai-secret-key"

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

camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# =====================================================
# SHARED DATA & STREAM STATE
# =====================================================

latest_frame = None
latest_id_boxes = []
latest_person_boxes = []
latest_id_card = False
frame_lock = threading.Lock()

# Flag to track if the detect page / video stream is active
is_streaming = False
stream_lock = threading.Lock()


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
    global latest_frame

    while True:
        success, frame = camera.read()

        if success:
            with frame_lock:
                latest_frame = frame
        else:
            print("Camera read failed")
            time.sleep(0.05)


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

        if not active:
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


@app.route("/detect")
def detect():
    return render_template("detect.html")


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == "admin" and password == "admin123":
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
        person_id_filter=person_id_filter
    )
@app.route("/admin/history")
def admin_history():
    filter_date = request.args.get("date", "")
    person_id = request.args.get("person_id", "")
    sort_order = request.args.get("sort", "latest")
    
    violations = get_filtered_violations(filter_date=filter_date, sort_order=sort_order, person_id=person_id)
    return render_template("admin_violation_history.html", violations=violations, filter_date=filter_date, person_id=person_id)

@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect("/")


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


@app.route("/api/status")
def api_status():
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

def generate_frames():
    global is_streaming, latest_person_boxes, latest_id_boxes
    
    with stream_lock:
        is_streaming = True

    try:
        while True:
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
            is_streaming = False
            with frame_lock:
                latest_person_boxes = []
                latest_id_boxes = []


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/result")
def result():
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