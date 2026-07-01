import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np

def download_and_prepare():
    print("Starting dataset preparation...")
    
    # Path settings
    os.makedirs('dataset', exist_ok=True)
    zip_path = 'dataset/student.zip'
    csv_path = 'dataset/student-mat.csv'
    output_path = 'dataset/student_performance.csv'
    
    # Step 1: Download UCI Student Performance dataset if not exists
    if not os.path.exists(csv_path):
        print("Downloading raw UCI Student Performance dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
        try:
            urllib.request.urlretrieve(url, zip_path)
            print("Download complete. Extracting student-mat.csv...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract('student-mat.csv', 'dataset')
            os.remove(zip_path)
            print("Extraction complete.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please ensure you have an active internet connection.")
            return False
    else:
        print("Raw dataset student-mat.csv already exists.")
        
    # Step 2: Read raw dataset
    df_raw = pd.read_csv(csv_path, sep=';')
    print(f"Loaded raw data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")
    
    # Set seed for reproducibility
    np.random.seed(42)
    n_samples = len(df_raw)
    
    # Step 3: Map and construct the student performance tracker features
    df = pd.DataFrame()
    
    # Gender
    df['Gender'] = df_raw['sex'].map({'F': 'Female', 'M': 'Male'})
    
    # Age
    df['Age'] = df_raw['age']
    
    # Study Hours: convert studytime category (1-4) to continuous hours
    # 1: <2h (map to 1-2h), 2: 2-5h (map to 2-5h), 3: 5-10h (map to 5-10h), 4: >10h (map to 10-16h)
    study_ranges = {
        1: (1.0, 2.0),
        2: (2.0, 5.0),
        3: (5.0, 10.0),
        4: (10.0, 16.0)
    }
    study_hours = []
    for st in df_raw['studytime']:
        low, high = study_ranges.get(st, (2.0, 5.0))
        study_hours.append(np.round(np.random.uniform(low, high), 1))
    df['Study_Hours'] = study_hours
    
    # Attendance: absences mapped to an attendance percentage out of 100%
    # max absences in dataset is 75 (GP school has up to 75, MS up to 93 in total Portuguese but here Math is 75)
    max_abs = df_raw['absences'].max() if df_raw['absences'].max() > 0 else 75
    # Calculate Attendance = (1 - absences / max_absences) * 100
    # Let's use a standard 93 as total classes to prevent 0% attendance being common, or use a custom formula:
    # Attendance = max(0.0, 100.0 - absences * 1.5)
    df['Attendance'] = np.round(np.maximum(0.0, 100.0 - df_raw['absences'] * 1.5), 1)
    
    # Previous Score: G1 scaled from 0-20 to 0-100 (multiply by 5)
    df['Previous_Score'] = df_raw['G1'] * 5.0
    
    # Parent Education: max of Medu and Fedu, mapped to categories
    # Medu: mother education (0-none, 1-primary, 2-5-9th grade, 3-secondary, 4-higher)
    # Fedu: father education
    parent_edu_numeric = np.maximum(df_raw['Medu'], df_raw['Fedu'])
    edu_map = {
        0: 'None',
        1: 'Primary Education',
        2: '5th to 9th Grade',
        3: 'Secondary Education',
        4: 'Higher Education'
    }
    df['Parent_Education'] = parent_edu_numeric.map(edu_map)
    
    # Internet Access
    df['Internet_Access'] = df_raw['internet'].map({'yes': 'Yes', 'no': 'No'})
    
    # Extra Activities
    df['Extra_Activities'] = df_raw['activities'].map({'yes': 'Yes', 'no': 'No'})
    
    # Family Support
    df['Family_Support'] = df_raw['famsup'].map({'yes': 'Yes', 'no': 'No'})
    
    # Sleep Hours: generate realistic values slightly correlated with goout and studytime
    # standard sleep of 7.5 hours, minus some factor of going out and studying
    sleep = 7.5 - (df_raw['goout'] - 3) * 0.4 - (df_raw['studytime'] - 2) * 0.25 + np.random.normal(0, 0.6, n_samples)
    df['Sleep_Hours'] = np.round(np.clip(sleep, 4.0, 10.0), 1)
    
    # Target Variable: Final Score (G3 scaled from 0-20 to 0-100)
    df['Final_Score'] = df_raw['G3'] * 5.0
    
    # Step 4: Inject Duplicates
    # Let's clone 5 random rows and append them
    dup_indices = np.random.choice(df.index, size=5, replace=False)
    df_duplicates = df.loc[dup_indices].copy()
    df = pd.concat([df, df_duplicates], ignore_index=True)
    print(f"Injected {len(dup_indices)} duplicate rows.")
    
    # Step 5: Inject Missing Values (~2% randomly across specific columns)
    # We inject into: Attendance, Study_Hours, Sleep_Hours, Parent_Education
    columns_with_nans = ['Attendance', 'Study_Hours', 'Sleep_Hours', 'Parent_Education']
    for col in columns_with_nans:
        nan_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
        df.loc[nan_indices, col] = np.nan
    print("Injected ~2% missing values (NaNs) in Attendance, Study_Hours, Sleep_Hours, and Parent_Education.")
    
    # Save the prepared dataset
    df.to_csv(output_path, index=False)
    print(f"Dataset prepared successfully and saved to: {output_path}")
    print(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    
if __name__ == '__main__':
    download_and_prepare()
