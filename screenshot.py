import cv2
import os
from datetime import datetime


def save_violation_screenshot(
    frame,
    person,
    person_id,
    violation_type
):
    """
    Save one screenshot for a person's violation.

    Folder structure:

    static/
        violations/
            person_1/
                2026-10-07/
                    ID_Card.jpg
    """

    date = datetime.now().strftime("%Y-%m-%d")

    safe_violation = (
        violation_type
        .replace(" ", "_")
        .replace("/", "_")
    )

    folder = os.path.join(
        "static",
        "violations",
        f"person_{person_id}",
        date
    )

    os.makedirs(
        folder,
        exist_ok=True
    )

    filename = f"{safe_violation}.jpg"

    full_path = os.path.join(
        folder,
        filename
    )

    # -----------------------------------------
    # PERSON CROP
    # -----------------------------------------

    x1 = max(0, int(person["x1"]))
    y1 = max(0, int(person["y1"]))

    x2 = min(
        frame.shape[1],
        int(person["x2"])
    )

    y2 = min(
        frame.shape[0],
        int(person["y2"])
    )

    person_image = frame[
        y1:y2,
        x1:x2
    ]

    # -----------------------------------------
    # SAFETY CHECK
    # -----------------------------------------

    if person_image.size == 0:
        print(
            "⚠️ Could not create person screenshot"
        )

        return None

    # -----------------------------------------
    # SAVE
    # -----------------------------------------

    success = cv2.imwrite(
        full_path,
        person_image
    )

    if not success:

        print(
            "❌ Failed to save screenshot"
        )

        return None

    print(
        f"📸 Screenshot saved: {full_path}"
    )

    # Store path using forward slashes
    # so Flask can serve it correctly.

    relative_path = full_path.replace(
        "\\",
        "/"
    )

    return relative_path