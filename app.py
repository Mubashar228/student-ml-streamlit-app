import streamlit as st
import sqlite3
import pandas as pd
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Tuple

# ML imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

DB_PATH = "student_app.db"

# -----------------------------
# Database Utilities
# -----------------------------

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                contact TEXT,
                class_name TEXT,
                previous_result TEXT CHECK(previous_result IN ('A','B','C','D','F'))
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_routine (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                shoes TEXT CHECK(shoes IN ('Yes','No')) NOT NULL,
                uniform TEXT CHECK(uniform IN ('Yes','No')) NOT NULL,
                attendance TEXT CHECK(attendance IN ('Present','Absent','Leave')) NOT NULL,
                notes TEXT,
                FOREIGN KEY(student_id) REFERENCES students(id)
            )
            """
        )
        conn.commit()


# -----------------------------
# Seed Helper (optional)
# -----------------------------

def seed_example_data():
    with get_conn() as conn:
        df_students = pd.read_sql("SELECT * FROM students", conn)
        if df_students.empty:
            conn.executemany(
                "INSERT INTO students(name, contact, class_name, previous_result) VALUES (?,?,?,?)",
                [
                    ("Ali", "03001234567", "Grade 5", "B"),
                    ("Sara", "03007654321", "Grade 5", "A"),
                    ("Hassan", "03001112233", "Grade 4", "C"),
                ],
            )
            conn.commit()
        df_r = pd.read_sql("SELECT * FROM daily_routine", conn)
        if df_r.empty:
            today = date.today().isoformat()
            rows = [
                (1, today, "Yes", "Yes", "Present", "Good start"),
                (2, today, "No", "Yes", "Absent", "Sick"),
                (3, today, "Yes", "No", "Present", "Forgot tie"),
            ]
            conn.executemany(
                "INSERT INTO daily_routine(student_id, date, shoes, uniform, attendance, notes) VALUES (?,?,?,?,?,?)",
                rows,
            )
            conn.commit()


# -----------------------------
# ML Utilities
# -----------------------------

class SimpleGradeModel:
    """
    Retrains quickly from DB on demand. Target uses students.previous_result
    as a proxy label. When enough routine rows exist, we train; otherwise
    we fall back to a rule-of-thumb heuristic.
    """

    def __init__(self):
        self.clf: Optional[DecisionTreeClassifier] = None
        self.le_target: Optional[LabelEncoder] = None
        self.ready: bool = False
        self.train_rows: int = 0
        self.train_acc: Optional[float] = None

    @staticmethod
    def _encode_features(df: pd.DataFrame) -> pd.DataFrame:
        # Map categorical features to numeric
        map_yesno = {"Yes": 1, "No": 0}
        map_att = {"Present": 2, "Leave": 1, "Absent": 0}
        df = df.copy()
        df["shoes_num"] = df["shoes"].map(map_yesno).fillna(0)
        df["uniform_num"] = df["uniform"].map(map_yesno).fillna(0)
        df["attendance_num"] = df["attendance"].map(map_att).fillna(0)
        return df

    def retrain(self, conn) -> None:
        df = pd.read_sql(
            """
            SELECT dr.*, s.previous_result
            FROM daily_routine dr
            JOIN students s ON s.id = dr.student_id
            """,
            conn,
        )
        if df.empty:
            self.ready = False
            self.train_rows = 0
            return
        df = self._encode_features(df)
        X = df[["shoes_num", "uniform_num", "attendance_num"]]
        y = df["previous_result"].astype(str)
        # Need at least 15 rows to be minimally meaningful
        if len(df) < 15:
            self.ready = False
            self.train_rows = len(df)
            return
        self.le_target = LabelEncoder()
        y_enc = self.le_target.fit_transform(y)
        clf = DecisionTreeClassifier(random_state=42, max_depth=4)
        clf.fit(X, y_enc)
        # quick resubstitution accuracy for info only
        y_hat = clf.predict(X)
        acc = accuracy_score(y_enc, y_hat)
        self.clf = clf
        self.train_acc = float(acc)
        self.ready = True
        self.train_rows = len(df)

    def predict_grade(self, shoes: str, uniform: str, attendance: str) -> Tuple[str, str]:
        """Returns (grade, info) as tuple.
        If model not ready, uses heuristic.
        """
        # Heuristic fallback
        def heuristic():
            score = 0
            if shoes == "Yes":
                score += 1
            if uniform == "Yes":
                score += 1
            score += {"Present": 2, "Leave": 1, "Absent": 0}.get(attendance, 0)
            # Map score 0-4 to grades
            if score >= 4:
                return "A"
            if score == 3:
                return "B"
            if score == 2:
                return "C"
            if score == 1:
                return "D"
            return "F"

        if not self.ready or self.clf is None or self.le_target is None:
            g = heuristic()
            return g, "Heuristic (model will train automatically when enough data is available)."
        # model path
        X = np.array(
            [
                [
                    1 if shoes == "Yes" else 0,
                    1 if uniform == "Yes" else 0,
                    {"Present": 2, "Leave": 1, "Absent": 0}.get(attendance, 0),
                ]
            ]
        )
        y_hat = self.clf.predict(X)
        grade = self.le_target.inverse_transform(y_hat)[0]
        return grade, f"ML model (trained on {self.train_rows} rows, approx. acc {self.train_acc:.2f})."


# -----------------------------
# Streamlit App
# -----------------------------

st.set_page_config(page_title="Student Performance & Routine (ML)", page_icon="🎓", layout="wide")

# Initialize DB and (optionally) seed
init_db()
if st.sidebar.checkbox("Load demo data (optional)"):
    seed_example_data()

# Build/retrain model on demand
model = SimpleGradeModel()
with get_conn() as _conn:
    model.retrain(_conn)

st.title("🎓 Student Performance & Daily Routine — ML Web App")
st.caption(
    "Add students, record daily routine (shoes, uniform, attendance), and get an instant predicted grade."
)

# Tabs for UX
tab_add_student, tab_add_routine, tab_dashboard, tab_data = st.tabs(
    ["➕ Add Student", "📝 Daily Entry", "📊 Dashboard", "🗂️ Data"]
)

# -----------------------------
# Add Student Tab
# -----------------------------
with tab_add_student:
    st.subheader("Add a new student")
    with st.form("add_student_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("Full Name", placeholder="e.g., Ali Raza", max_chars=80)
        with col2:
            contact = st.text_input("Contact Number", placeholder="03XXXXXXXXX", max_chars=20)
        with col3:
            class_name = st.text_input("Class/Grade", placeholder="e.g., Grade 5", max_chars=30)
        prev = st.selectbox("Previous Result (grade)", ["A", "B", "C", "D", "F"], index=1)
        submitted = st.form_submit_button("Add Student")
        if submitted:
            if not name.strip():
                st.error("Name is required.")
            else:
                with get_conn() as conn:
                    conn.execute(
                        "INSERT INTO students(name, contact, class_name, previous_result) VALUES (?,?,?,?)",
                        (name.strip(), contact.strip(), class_name.strip(), prev),
                    )
                    conn.commit()
                st.success(f"Student '{name}' added successfully!")

# -----------------------------
# Daily Entry Tab
# -----------------------------
with tab_add_routine:
    st.subheader("Record today's routine & get prediction")
    # Load students for selection
    with get_conn() as conn:
        students_df = pd.read_sql("SELECT id, name, class_name FROM students ORDER BY name", conn)
    if students_df.empty:
        st.info("No students found. Please add a student first.")
    else:
        students_df["label"] = students_df.apply(
            lambda r: f"{r['name']} (ID {r['id']}) — {r['class_name']}", axis=1
        )
        selected = st.selectbox(
            "Select Student",
            options=students_df["id"].tolist(),
            format_func=lambda sid: students_df.loc[students_df.id == sid, "label"].values[0],
        )
        today_str = date.today().isoformat()
        with st.form("routine_form", clear_on_submit=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                rdate = st.date_input("Date", value=date.today())
            with c2:
                shoes = st.radio("Shoes", ["Yes", "No"], horizontal=True)
            with c3:
                uniform = st.radio("Uniform", ["Yes", "No"], horizontal=True)
            with c4:
                attendance = st.selectbox("Attendance", ["Present", "Absent", "Leave"], index=0)
            notes = st.text_input("Notes (optional)", placeholder="e.g., Late by 10 min")
            save = st.form_submit_button("Save & Predict")
        if save:
            with get_conn() as conn:
                conn.execute(
                    "INSERT INTO daily_routine(student_id, date, shoes, uniform, attendance, notes) VALUES (?,?,?,?,?,?)",
                    (int(selected), rdate.isoformat(), shoes, uniform, attendance, notes),
                )
                conn.commit()
            # retrain after new data
            with get_conn() as _c:
                model.retrain(_c)
            grade, info = model.predict_grade(shoes, uniform, attendance)
            st.success(f"Predicted Grade: {grade}")
            st.caption(info)

# -----------------------------
# Dashboard Tab
# -----------------------------
with tab_dashboard:
    st.subheader("Student Dashboard")
    with get_conn() as conn:
        df_students = pd.read_sql(
            "SELECT id, name, contact, class_name, previous_result FROM students ORDER BY name",
            conn,
        )
    if df_students.empty:
        st.info("No students yet.")
    else:
        sid = st.selectbox(
            "Choose a student",
            options=df_students["id"].tolist(),
            format_func=lambda s: df_students.loc[df_students.id == s, "name"].values[0],
        )
        with get_conn() as conn:
            df_profile = df_students[df_students.id == sid].copy()
            df_r = pd.read_sql(
                "SELECT date, shoes, uniform, attendance, notes FROM daily_routine WHERE student_id = ? ORDER BY date DESC",
                conn,
                params=(int(sid),),
            )
        st.write("### Profile")
        p = df_profile.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Name", p["name"])
        c2.metric("Class", p["class_name"])
        c3.metric("Contact", p["contact"] if p["contact"] else "—")
        st.metric("Previous Result", p["previous_result"])  # single line metric

        st.write("### Recent Routine Records")
        if df_r.empty:
            st.warning("No routine records yet for this student.")
        else:
            st.dataframe(df_r, use_container_width=True)
            # Compute latest prediction from most recent row
            last = df_r.iloc[0]
            g, info = model.predict_grade(
                shoes=last["shoes"], uniform=last["uniform"], attendance=last["attendance"]
            )
            st.success(f"Latest Predicted Grade (based on {last['date']}): {g}")
            st.caption(info)

            # Attendance summary (last 30)
            last30 = df_r.head(30)
            att_counts = last30["attendance"].value_counts()
            p_rate = att_counts.get("Present", 0) / max(1, len(last30))
            st.write(f"**30-day present rate:** {p_rate:.0%}")

# -----------------------------
# Data Tab
# -----------------------------
with tab_data:
    st.subheader("All Data (Export-ready)")
    colA, colB = st.columns(2)
    with get_conn() as conn:
        all_students = pd.read_sql("SELECT * FROM students ORDER BY id", conn)
        all_routines = pd.read_sql("SELECT * FROM daily_routine ORDER BY date DESC", conn)
    with colA:
        st.write("**Students**")
        st.dataframe(all_students, use_container_width=True)
        if st.button("Download Students CSV"):
            all_students.to_csv("students_export.csv", index=False)
            st.success("Saved as students_export.csv in the app directory.")
    with colB:
        st.write("**Daily Routine**")
        st.dataframe(all_routines, use_container_width=True)
        if st.button("Download Routine CSV"):
            all_routines.to_csv("routines_export.csv", index=False)
            st.success("Saved as routines_export.csv in the app directory.")

# Footer note
st.caption(
    "Tip: The ML model trains automatically when 15+ routine records exist. Before that, a simple heuristic is used for predictions."
)
