import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_synthetic_df(n=40):
    import pandas as pd

    return pd.DataFrame(
        {
            "study_time_hours": [float(i % 10 + 1) for i in range(n)],
            "attendance_percent": [float(60 + i % 40) for i in range(n)],
            "sleep_hours": [float(5 + i % 4) for i in range(n)],
            "previous_grade": [float(50 + i % 50) for i in range(n)],
            "gender": ["Male" if i % 2 == 0 else "Female" for i in range(n)],
            "parental_education": ["Bachelors" if i % 3 else "None" for i in range(n)],
            "internet_access": ["Yes" if i % 2 else "No" for i in range(n)],
            "extracurricular_activities": ["Yes" if i % 4 else "No" for i in range(n)],
            "part_time_job": ["No" if i % 5 else "Yes" for i in range(n)],
        }
    )
