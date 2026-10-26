import sqlite3
from datetime import datetime


DATABASE = "campus_appearance.db"


def get_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            violation_type TEXT NOT NULL,
            violation_date TEXT NOT NULL,
            count INTEGER DEFAULT 0,
            first_detected TEXT,
            last_detected TEXT,
            screenshot_path TEXT,
            assigned_person_id INTEGER,
            created_at TEXT NOT NULL
        )
    """)

    try:
        conn.execute("ALTER TABLE violations ADD COLUMN assigned_person_id INTEGER")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()


def get_today():
    return datetime.now().strftime("%Y-%m-%d")


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def find_violation(person_id, violation_type, date):
    conn = get_connection()

    row = conn.execute("""
        SELECT *
        FROM violations
        WHERE person_id = ?
        AND violation_type = ?
        AND violation_date = ?
    """, (
        person_id,
        violation_type,
        date
    )).fetchone()

    conn.close()
    return row


def create_violation(person_id, violation_type, screenshot_path=None):
    now = get_current_time()
    date = get_today()

    conn = get_connection()

    cursor = conn.execute("""
        INSERT INTO violations (
            person_id,
            violation_type,
            violation_date,
            count,
            first_detected,
            last_detected,
            screenshot_path,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        person_id,
        violation_type,
        date,
        1,
        now,
        now,
        screenshot_path,
        now
    ))

    conn.commit()
    violation_id = cursor.lastrowid
    conn.close()

    return violation_id


def update_violation_count(violation_id):
    now = get_current_time()

    conn = get_connection()

    conn.execute("""
        UPDATE violations
        SET
            count = count + 1,
            last_detected = ?
        WHERE id = ?
    """, (
        now,
        violation_id
    ))

    conn.commit()
    conn.close()


def update_screenshot(violation_id, screenshot_path):
    conn = get_connection()

    conn.execute("""
        UPDATE violations
        SET screenshot_path = ?
        WHERE id = ?
    """, (
        screenshot_path,
        violation_id
    ))

    conn.commit()
    conn.close()


def get_all_people():
    conn = get_connection()

    rows = conn.execute("""
        SELECT
            v.person_id,
            MAX(v.last_detected) AS last_detected,
            COUNT(*) AS violation_types,
            (
                SELECT screenshot_path
                FROM violations
                WHERE person_id = v.person_id
                ORDER BY last_detected DESC, id DESC
                LIMIT 1
            ) AS screenshot_path
        FROM violations v
        GROUP BY v.person_id
        ORDER BY v.person_id
    """).fetchall()

    conn.close()
    return rows


def get_filtered_violations(filter_date="", sort_order="latest", person_id=""):
    conn = get_connection()
    try:
        query = """
            SELECT
                v.*,
                p.person_code AS assigned_person_code,
                p.display_name AS assigned_person_name
            FROM violations v
            LEFT JOIN people p ON p.id = v.assigned_person_id
            WHERE v.violation_type = 'ID Card'
        """
        params = []

        if filter_date:
            query += " AND v.violation_date = ?"
            params.append(filter_date)

        if person_id:
            query += " AND v.person_id = ?"
            params.append(person_id)

        if sort_order == "oldest":
            query += " ORDER BY v.last_detected ASC, v.id ASC"
        else:
            query += " ORDER BY v.last_detected DESC, v.id DESC"

        return conn.execute(query, params).fetchall()
    finally:
        conn.close()