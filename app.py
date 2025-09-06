import streamlit as st
import sqlite3
import pandas as pd
import os
import io
from datetime import date, timedelta
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from typing import Optional, Tuple

# ---------------- CONFIG ----------------
DB_PATH = "student_app.db"
EXPORT_DIR = "exports"
MIN_ROWS_FOR_MODEL = 15

# ---------------- DB HELPERS ----------------
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


# ---------------- DEMO SEED ----------------
def seed_demo_data():
    with get_conn() as conn:
        students = pd.read_sql("SELECT * FROM students", conn)
        if students.empty:
            demo_students = [
                ("Ali", "03001234567", "Grade 5", "B"),
                ("Sara", "03007654321", "Grade 5", "A"),
                ("Hassan", "03001112233", "Grade 4", "C"),
                ("Fatima", "03009876543", "Grade 6", "A"),
                ("Zain", "03005554433", "Grade 5", "D"),
            ]
            conn.executemany(
                "INSERT INTO students(name,contact,class_name,previous_result) VALUES (?,?,?,?)",
                demo_students,
            )
            conn.commit()
        routines = pd.read_sql("SELECT * FROM daily_routine", conn)
        if routines.empty:
            today = date.today()
            rows = []
            with get_conn() as c:
                s = pd.read_sql("SELECT id FROM students", c)
            for sid in s["id"].tolist():
                for i in range(10):
                    d = (today - timedelta(days=i)).isoformat()
                    rows.append((int(sid), d, "Yes" if sid % 2 == 0 else "No", "Yes", "Present" if i % 3 != 0 else "Absent", ""))
            with get_conn() as conn:
                conn.executemany(
                    "INSERT INTO daily_routine(student_id, date, shoes, uniform, attendance, notes) VALUES (?,?,?,?,?,?)",
                    rows,
                )
                conn.commit()


# ---------------- ML Model ----------------
class GradeModel:
    def __init__(self, min_rows: int = MIN_ROWS_FOR_MODEL):
        self.clf: Optional[DecisionTreeClassifier] = None
        self.le: Optional[LabelEncoder] = None
        self.ready = False
        self.train_rows = 0
        self.min_rows = min_rows
        self.train_acc = None

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["shoes_num"] = df["shoes"].map({"Yes": 1, "No": 0}).fillna(0)
        df["uniform_num"] = df["uniform"].map({"Yes": 1, "No": 0}).fillna(0)
        df["attendance_num"] = df["attendance"].map({"Present": 2, "Leave": 1, "Absent": 0}).fillna(0)
        return df

    def train(self):
        with get_conn() as conn:
            try:
                df = pd.read_sql(
                    "SELECT dr.*, s.previous_result FROM daily_routine dr JOIN students s ON s.id = dr.student_id",
                    conn,
                    parse_dates=["date"],
                )
            except Exception:
                self.ready = False
                return
        if df.empty or len(df) < self.min_rows:
            self.ready = False
            self.train_rows = len(df) if not df.empty else 0
            return
        df = self._encode(df)
        X = df[["shoes_num", "uniform_num", "attendance_num"]]
        y = df["previous_result"].astype(str)
        self.le = LabelEncoder()
        y_enc = self.le.fit_transform(y)
        clf = DecisionTreeClassifier(random_state=42, max_depth=4)
        clf.fit(X, y_enc)
        # store
        self.clf = clf
        self.train_rows = len(df)
        pred = clf.predict(X)
        self.train_acc = accuracy_score(y_enc, pred)
        self.ready = True

    def predict(self, shoes: str, uniform: str, attendance: str) -> Tuple[str, str]:
        # heuristic fallback
        def heuristic():
            score = 0
            if shoes == "Yes":
                score += 1
            if uniform == "Yes":
                score += 1
            score += {"Present": 2, "Leave": 1, "Absent": 0}.get(attendance, 0)
            if score >= 4:
                return "A"
            if score == 3:
                return "B"
            if score == 2:
                return "C"
            if score == 1:
                return "D"
            return "F"

        if not self.ready or self.clf is None or self.le is None:
            return heuristic(), f"Heuristic used — model trains when {self.min_rows}+ rows exist (current {self.train_rows})."
        X = np.array([[1 if shoes == "Yes" else 0, 1 if uniform == "Yes" else 0, {"Present": 2, "Leave": 1, "Absent": 0}.get(attendance, 0)]])
        yhat = self.clf.predict(X)
        grade = self.le.inverse_transform(yhat)[0]
        return grade, f"ML model (trained on {self.train_rows} rows, acc {self.train_acc:.2f})."


# ---------------- HELPERS ----------------
def attendance_percent(df_r: pd.DataFrame) -> float:
    if df_r.empty:
        return 0.0
    return (df_r["attendance"] == "Present").sum() / len(df_r)


def compliance_percent(df_r: pd.DataFrame, col: str) -> float:
    if df_r.empty:
        return 0.0
    return (df_r[col] == "Yes").sum() / len(df_r)


def compute_student_score(df_r: pd.DataFrame, prev_grade: str) -> float:
    if df_r.empty:
        return 0.0
    att = (df_r["attendance"] == "Present").sum() / len(df_r)
    comp = ((df_r["shoes"] == "Yes").sum() + (df_r["uniform"] == "Yes").sum()) / (2 * len(df_r))
    grade_map = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4, "F": 0.2}
    g = grade_map.get(prev_grade, 0.5)
    return 0.6 * att + 0.2 * comp + 0.2 * g


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def export_class_reports(class_name: str, start_date: date, end_date: date) -> dict:
    """Creates exports in exports/Class_<class_name>/ and returns dict of {filename: bytes} for downloads."""
    out = {}
    with get_conn() as conn:
        students = pd.read_sql("SELECT * FROM students WHERE class_name = ?", conn, params=(class_name,))
        routines = pd.read_sql("SELECT * FROM daily_routine WHERE date BETWEEN ? AND ?", conn, params=(start_date.isoformat(), end_date.isoformat()))
        # filter routines of this class
        if not students.empty and not routines.empty:
            routines = routines[routines["student_id"].isin(students["id"].tolist())]
    folder = os.path.join(EXPORT_DIR, f"Class_{class_name}")
    ensure_dir(folder)

    # save students
    if 'students' in locals() and not students.empty:
        fname = os.path.join(folder, f"students_{class_name}.csv")
        students.to_csv(fname, index=False)
        out[os.path.basename(fname)] = to_csv_bytes(students)
    # save routines
    if 'routines' in locals() and not routines.empty:
        fname = os.path.join(folder, f"routines_{class_name}_{start_date.isoformat()}_{end_date.isoformat()}.csv")
        routines.to_csv(fname, index=False)
        out[os.path.basename(fname)] = to_csv_bytes(routines)

    # Top/Bottom 5 in period
    if 'routines' in locals() and not routines.empty and 'students' in locals() and not students.empty:
        merged = routines.merge(students, left_on="student_id", right_on="id")
        scores = []
        for sid, grp in merged.groupby("student_id"):
            prev = grp["previous_result"].iloc[0]
            nm = grp["name"].iloc[0]
            sc = compute_student_score(grp, prev)
            scores.append((sid, nm, sc))
        df_scores = pd.DataFrame(scores, columns=["id", "name", "score"]).sort_values("score", ascending=False)
        top = df_scores.head(5)
        bottom = df_scores.tail(5)
        if not top.empty:
            fname = os.path.join(folder, f"top5_{class_name}_{start_date.isoformat()}_{end_date.isoformat()}.csv")
            top.to_csv(fname, index=False)
            out[os.path.basename(fname)] = to_csv_bytes(top)
        if not bottom.empty:
            fname = os.path.join(folder, f"bottom5_{class_name}_{start_date.isoformat()}_{end_date.isoformat()}.csv")
            bottom.to_csv(fname, index=False)
            out[os.path.basename(fname)] = to_csv_bytes(bottom)

    return out


# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="Govt Model Elementary School Chak.243 JB", page_icon="🎓", layout="wide")
init_db()
model = GradeModel()
model.train()

# Sidebar actions
st.sidebar.header("Utilities")
if st.sidebar.button("Load demo data"):
    seed_demo_data()
    model.train()
    st.sidebar.success("Demo data loaded.")

# Main UI
st.title("🎓 Govt Model Elementary School Chak.243 JB — Student Routine & ML App")
st.write("Add students, record daily routine, view class-wise analytics, and export class reports.")

tabs = st.tabs(["➕ Add Student", "⚡ Quick Entry", "📝 Full Entry", "📈 Analytics", "📊 Dashboard", "🗂️ Data & Export"])

# ----------- Add Student -----------
with tabs[0]:
    st.header("Add New Student")
    with st.form("add_student", clear_on_submit=True):
        name = st.text_input("Full name")
        contact = st.text_input("Contact number")
        class_name = st.text_input("Class / Grade (e.g., Grade 5)")
        prev = st.selectbox("Previous Result (grade)", ["A", "B", "C", "D", "F"], index=1)
        if st.form_submit_button("Add Student"):
            if not name.strip():
                st.error("Name is required.")
            else:
                with get_conn() as conn:
                    conn.execute(
                        "INSERT INTO students(name, contact, class_name, previous_result) VALUES (?,?,?,?)",
                        (name.strip(), contact.strip(), class_name.strip(), prev),
                    )
                    conn.commit()
                model.train()
                st.success(f"Student '{name}' added.")

# ----------- Quick Entry -----------
with tabs[1]:
    st.header("Quick Daily Entry (tick per student)")
    with get_conn() as conn:
        students = pd.read_sql("SELECT * FROM students ORDER BY class_name,name", conn)
    if students.empty:
        st.info("No students found. Add students first.")
    else:
        # class filter for quick entry
        classes = ["All Classes"] + sorted(students["class_name"].dropna().unique().tolist())
        sel_class = st.selectbox("Filter by class (optional)", classes)
        if sel_class != "All Classes":
            students = students[students["class_name"] == sel_class]
        entry_date = st.date_input("Date", value=date.today())
        with st.form("quick_entries"):
            rows = []
            for _, r in students.iterrows():
                cols = st.columns([3, 1, 1, 1])
                cols[0].write(f"**{r['name']}** — {r['class_name']}")
                shoes = cols[1].radio(f"q_sh_{r['id']}", ["Yes", "No"], index=0, key=f"q_sh_{r['id']}")
                uniform = cols[2].radio(f"q_un_{r['id']}", ["Yes", "No"], index=0, key=f"q_un_{r['id']}")
                attendance = cols[3].selectbox(f"q_at_{r['id']}", ["Present", "Absent", "Leave"], key=f"q_at_{r['id']}")
                rows.append((int(r["id"]), entry_date.isoformat(), shoes, uniform, attendance, ""))
            if st.form_submit_button("Save All Quick Entries"):
                with get_conn() as conn:
                    conn.executemany(
                        "INSERT INTO daily_routine(student_id, date, shoes, uniform, attendance, notes) VALUES (?,?,?,?,?,?)",
                        rows,
                    )
                    conn.commit()
                model.train()
                st.success("Saved quick entries for selected students.")

# ----------- Full Entry -----------
with tabs[2]:
    st.header("Full Daily Entry (single student)")
    with get_conn() as conn:
        students = pd.read_sql("SELECT * FROM students ORDER BY class_name,name", conn)
    if students.empty:
        st.info("No students to enter. Add students first.")
    else:
        sid = st.selectbox("Select student", options=students["id"].tolist(), format_func=lambda x: students.loc[students.id == x, "name"].values[0])
        with st.form("full_entry"):
            rdate = st.date_input("Date", value=date.today())
            shoes = st.radio("Shoes", ["Yes", "No"], index=0)
            uniform = st.radio("Uniform", ["Yes", "No"], index=0)
            attendance = st.selectbox("Attendance", ["Present", "Absent", "Leave"], index=0)
            notes = st.text_input("Notes (optional)")
            if st.form_submit_button("Save Entry"):
                with get_conn() as conn:
                    conn.execute(
                        "INSERT INTO daily_routine(student_id, date, shoes, uniform, attendance, notes) VALUES (?,?,?,?,?,?)",
                        (int(sid), rdate.isoformat(), shoes, uniform, attendance, notes),
                    )
                    conn.commit()
                model.train()
                grade, info = model.predict(shoes, uniform, attendance)
                st.success(f"Saved. Predicted Grade: {grade}")
                st.info(info)

# ----------- Analytics -----------
with tabs[3]:
    st.header("Analytics — Class-wise (weekly / monthly / yearly)")
    range_key = st.selectbox("Select range", ["7 days (weekly)", "30 days (monthly)", "365 days (yearly)"])
    days_map = {"7 days (weekly)": 7, "30 days (monthly)": 30, "365 days (yearly)": 365}
    days = days_map[range_key]
    end = date.today()
    start = end - timedelta(days=days - 1)

    with get_conn() as conn:
        all_students = pd.read_sql("SELECT * FROM students ORDER BY class_name,name", conn)
        all_routines = pd.read_sql("SELECT * FROM daily_routine ORDER BY date DESC", conn, parse_dates=["date"])

    if all_students.empty:
        st.info("No students in database yet.")
    else:
        classes = ["All Classes"] + sorted(all_students["class_name"].dropna().unique().tolist())
        sel_class = st.selectbox("Select Class for analytics", classes)

        # Filter by class
        students = all_students.copy()
        routines = all_routines.copy()
        routines["date"] = pd.to_datetime(routines["date"]).dt.date
        if sel_class != "All Classes":
            students = students[students["class_name"] == sel_class]
            routines = routines[routines["student_id"].isin(students["id"].tolist())]

        period = routines[(routines["date"] >= start) & (routines["date"] <= end)]
        st.write(f"Analytics for **{sel_class}** from **{start.isoformat()}** to **{end.isoformat()}**")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Entries", len(period))
        col2.metric("Students in Class", students.shape[0])
        col3.metric("Present Rate", f"{attendance_percent(period):.0%}")
        col4.metric("Shoes Compliance", f"{compliance_percent(period,'shoes'):.0%}")
        st.metric("Uniform Compliance", f"{compliance_percent(period,'uniform'):.0%}")

        # Grade distribution from students table (class-wise)
        if not students.empty:
            grade_counts = students["previous_result"].value_counts().reindex(["A", "B", "C", "D", "F"]).fillna(0)
            fig, ax = plt.subplots()
            ax.bar(grade_counts.index.astype(str), grade_counts.values)
            ax.set_title("Grade distribution (previous_result)")
            st.pyplot(fig)

        # Attendance trend
        if not period.empty:
            daily = period.groupby("date")["attendance"].apply(lambda s: (s == "Present").sum() / s.count())
            st.line_chart(daily)

        # Top 5 / Bottom 5 in selected period
        if not students.empty and not period.empty:
            merged = period.merge(students, left_on="student_id", right_on="id")
            scores = []
            for sid, grp in merged.groupby("student_id"):
                prev = grp["previous_result"].iloc[0]
                name = grp["name"].iloc[0]
                sc = compute_student_score(grp, prev)
                scores.append({"id": sid, "name": name, "score": sc})
            df_scores = pd.DataFrame(scores).sort_values("score", ascending=False)
            st.write("Top 5 Students")
            st.table(df_scores.head(5).reset_index(drop=True))
            st.write("Bottom 5 Students")
            st.table(df_scores.tail(5).reset_index(drop=True))
        else:
            st.info("Not enough routine data in the selected range to compute Top/Bottom lists.")

        # Export class reports
        if sel_class != "All Classes":
            if st.button("Export reports for this class (save to exports folder)"):
                files = export_class_reports(sel_class, start, end)
                if not files:
                    st.warning("No data to export for this class / period.")
                else:
                    st.success(f"Exported {len(files)} files to {os.path.join(EXPORT_DIR, 'Class_' + sel_class)}")
                    for fname, content in files.items():
                        st.download_button(label=f"Download {fname}", data=content, file_name=fname, mime="text/csv")

# ----------- Dashboard -----------
with tabs[4]:
    st.header("Dashboard (All Classes or select class)")
    with get_conn() as conn:
        all_students = pd.read_sql("SELECT * FROM students ORDER BY class_name,name", conn)
        all_routines = pd.read_sql("SELECT * FROM daily_routine ORDER BY date DESC", conn, parse_dates=["date"])
    if all_students.empty or all_routines.empty:
        st.info("Add students and daily routines to view dashboard charts.")
    else:
        classes = ["All Classes"] + sorted(all_students["class_name"].dropna().unique().tolist())
        sel = st.selectbox("Filter dashboard by class", classes)
        students = all_students.copy()
        routines = all_routines.copy()
        routines["date"] = pd.to_datetime(routines["date"]).dt.date
        if sel != "All Classes":
            students = students[students["class_name"] == sel]
            routines = routines[routines["student_id"].isin(students["id"].tolist())]
        st.subheader(f"Summary — {sel}")

        # Attendance Pie
        att_counts = routines["attendance"].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(att_counts.values, labels=att_counts.index, autopct="%1.1f%%")
        ax1.set_title("Attendance Distribution")
        st.pyplot(fig1)

        # Compliance bar
        comp = {
            "Shoes": (routines["shoes"] == "Yes").mean() * 100 if not routines.empty else 0,
            "Uniform": (routines["uniform"] == "Yes").mean() * 100 if not routines.empty else 0,
        }
        fig2, ax2 = plt.subplots()
        ax2.bar(list(comp.keys()), list(comp.values()))
        ax2.set_ylabel("%")
        ax2.set_title("Shoes & Uniform Compliance")
        st.pyplot(fig2)

        # Grade distribution (students table)
        if not students.empty:
            grade_counts = students["previous_result"].value_counts().reindex(["A", "B", "C", "D", "F"]).fillna(0)
            fig3, ax3 = plt.subplots()
            ax3.bar(grade_counts.index.astype(str), grade_counts.values)
            ax3.set_title("Grade distribution (previous_result)")
            st.pyplot(fig3)

        # Top/Bottom
        if not routines.empty and not students.empty:
            merged = routines.merge(students, left_on="student_id", right_on="id")
            scores = []
            for sid, grp in merged.groupby("student_id"):
                prev = grp["previous_result"].iloc[0]
                name = grp["name"].iloc[0]
                sc = compute_student_score(grp, prev)
                scores.append({"name": name, "score": sc})
            df_scores = pd.DataFrame(scores).sort_values("score", ascending=False)
            st.write("Top 5 Students")
            st.table(df_scores.head(5).reset_index(drop=True))
            st.write("Bottom 5 Students")
            st.table(df_scores.tail(5).reset_index(drop=True))

# ----------- Data & Export -----------
with tabs[5]:
    st.header("Raw Data & Exports")
    with get_conn() as conn:
        students = pd.read_sql("SELECT * FROM students ORDER BY class_name,name", conn)
        routines = pd.read_sql("SELECT * FROM daily_routine ORDER BY date DESC", conn)
    st.subheader("Students")
    st.dataframe(students)
    if not students.empty:
        st.download_button("Download students CSV", to_csv_bytes(students), file_name="students.csv", mime="text/csv")

    st.subheader("Daily Routines")
    st.dataframe(routines)
    if not routines.empty:
        st.download_button("Download routines CSV", to_csv_bytes(routines), file_name="routines.csv", mime="text/csv")

    # Bulk export per class (all time or date-range)
    st.write("---")
    st.subheader("Bulk Export: Class-wise reports")
    classes = sorted(students["class_name"].dropna().unique().tolist()) if not students.empty else []
    if classes:
        exp_class = st.selectbox("Select class to export (or choose 'All classes' below)", ["All classes"] + classes)
        exp_range = st.selectbox("Export range", ["All time", "Last 30 days", "Last 7 days"])
        if st.button("Export and download"):
            start = date.min
            end = date.today()
            if exp_range == "Last 30 days":
                start = end - timedelta(days=29)
            elif exp_range == "Last 7 days":
                start = end - timedelta(days=6)

            files_created = {}
            if exp_class == "All classes":
                for cl in classes:
                    files = export_class_reports(cl, start, end)
                    files_created.update({f"{cl}/{k}": v for k, v in files.items()})
            else:
                files_created = export_class_reports(exp_class, start, end)

            if not files_created:
                st.warning("No files were created (maybe no data in the selected range).")
            else:
                st.success(f"Created {len(files_created)} files in '{EXPORT_DIR}' folder.")
                for fname, content in files_created.items():
                    st.download_button(label=f"Download {fname}", data=content, file_name=fname.replace('/', '_'), mime="text/csv")

st.caption("Tip: Use Quick Entry for fast daily marking. Analytics and exports are class-wise.")
