import pandas as pd
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'data')

INPUT_FILE  = os.path.join(DATA_DIR, 'college_placement_ds.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'cleaned_student_data.csv')

VALID_RANGES = {
    'CGPA'                : (0.0, 10.0),
    'Internships'         : (0,   10),
    'Projects'            : (0,   20),
    'Coding_Skills'       : (1,   10),
    'Communication_Skills': (1,   10),
    'Aptitude_Test_Score' : (0,   100),
    'Soft_Skills_Rating'  : (1,   10),
    'Certifications'      : (0,   20),
    'Backlogs'            : (0,   30),
}

GENDER_MAP    = {'Male': 1, 'Female': 0}
DEGREE_MAP    = {'B.Tech': 0, 'Bca': 1, 'Mca': 2, 'B.Sc': 3}
BRANCH_MAP    = {'Cse': 0, 'It': 1, 'Ece': 2, 'Me': 3, 'Civil': 4}
PLACEMENT_MAP = {'Placed': 1, 'Not Placed': 0}


def load_data():
    print("\nSTEP 1: Loading Raw Data")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")
    return df


def drop_irrelevant_columns(df):
    print("\nSTEP 2: Dropping Irrelevant Columns")
    cols_to_drop = ['Student_ID', 'Age']
    df = df.drop(columns=cols_to_drop)
    print(f"   Dropped: {cols_to_drop}")
    return df


def remove_duplicates(df):
    print("\nSTEP 3: Removing Duplicates")
    before = len(df)
    df = df.drop_duplicates()
    print(f"   Removed: {before - len(df)} duplicates")
    return df


def handle_missing_values(df):
    print("\nSTEP 4: Handling Missing Values")
    total_missing = df.isnull().sum().sum()
    if total_missing == 0:
        print("   No missing values found")
        return df

    numerical_cols = [
        'CGPA', 'Internships', 'Projects', 'Coding_Skills',
        'Communication_Skills', 'Aptitude_Test_Score',
        'Soft_Skills_Rating', 'Certifications', 'Backlogs'
    ]
    categorical_cols = ['Gender', 'Degree', 'Branch']

    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    df = df.dropna(subset=['Placement_Status'])
    print(f"   Missing values handled. Remaining: {df.isnull().sum().sum()}")
    return df


def clean_text_columns(df):
    print("\nSTEP 5: Cleaning Text Columns")
    text_cols = ['Gender', 'Degree', 'Branch', 'Placement_Status']
    for col in text_cols:
        df[col] = df[col].str.strip()
        df[col] = df[col].str.replace(r'[^a-zA-Z0-9\s.]', '', regex=True)
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        df[col] = df[col].str.strip()
        df[col] = df[col].str.title()
        print(f"   {col}: {sorted(df[col].unique().tolist())}")
    return df


def fix_outliers(df):
    print("\nSTEP 6: Fixing Outliers")
    for col, (lo, hi) in VALID_RANGES.items():
        out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
        df[col] = df[col].clip(lo, hi)
        print(f"   {col}: {out_of_range} outliers fixed")
    return df

def encode_categorical(df):
    print("\nSTEP 7: Encoding Categorical Columns")
    df['Gender_Encoded'] = df['Gender'].map(GENDER_MAP)
    df['Degree_Encoded'] = df['Degree'].map(DEGREE_MAP).fillna(-1)
    df['Branch_Encoded'] = df['Branch'].map(BRANCH_MAP).fillna(-1)
    print("   Gender_Encoded, Degree_Encoded, Branch_Encoded added")
    return df


def encode_target_label(df):
    print("\nSTEP 8: Encoding Target Label")
    df['Placement_Label'] = df['Placement_Status'].map(PLACEMENT_MAP)
    placed     = int(df['Placement_Label'].sum())
    not_placed = int((df['Placement_Label'] == 0).sum())
    print(f"   Placed: {placed:,}  |  Not Placed: {not_placed:,}")
    return df


def save_cleaned_data(df):
    print("\nSTEP 9: Saving Cleaned Data")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"   Saved: {OUTPUT_FILE}")
    print(f"   Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")


def main():
    print("STUDENT DATASET PREPROCESSING")

    df = load_data()
    df = drop_irrelevant_columns(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = clean_text_columns(df)
    df = fix_outliers(df)
    df = encode_categorical(df)
    df = encode_target_label(df)
    save_cleaned_data(df)

    print("\nPREPROCESSING COMPLETE")
    print(f"Output: data/cleaned_student_data.csv")

    return df


if __name__ == "__main__":
    main()
