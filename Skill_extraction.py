import pandas as pd
import os
import re
from collections import defaultdict

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, 'data')

AI_DATASET      = os.path.join(DATA_DIR, 'ai_job_dataset.csv')
JOB_DATASET     = os.path.join(DATA_DIR, 'job_descriptions.csv')
OUTPUT_FILE     = os.path.join(DATA_DIR, 'skills_per_role.csv')


RELEVANT_ROLES_JOB_DESC = {
    # IT & Software
    'Software Engineer', 'Software Developer', 'Software Architect',
    'Web Developer', 'Front-End Developer', 'Back-End Developer',
    'Front-End Engineer', 'UI Developer', 'UX/UI Designer',
    'Data Analyst', 'Data Scientist', 'Data Engineer',
    'Database Administrator', 'Database Developer',
    'Network Engineer', 'Network Administrator', 'Network Security Specialist',
    'Systems Engineer', 'Systems Administrator', 'Systems Analyst',
    'IT Support Specialist', 'IT Manager', 'IT Administrator',
    'QA Analyst', 'QA Engineer', 'Software Tester',
    'Java Developer', 'Business Analyst', 'Research Scientist',
    'Technical Writer',
    # Finance & Accounting
    'Financial Analyst', 'Financial Advisor', 'Financial Planner',
    'Financial Controller', 'Investment Analyst', 'Investment Banker',
    'Investment Advisor', 'Accountant', 'Tax Consultant',
    'Business Development Manager', 'Finance Manager',
    # HR & Operations
    'HR Manager', 'HR Coordinator', 'HR Generalist',
    'Human Resources Manager', 'Operations Manager',
    'Procurement Manager', 'Procurement Specialist', 'Procurement Coordinator',
    'Supply Chain Manager', 'Supply Chain Analyst',
    'Project Manager', 'Project Coordinator',
    'Office Manager', 'Administrative Assistant',
}


NORMALIZATION_MAP = {
    'nlp'                    : 'natural language processing',
    'ml'                     : 'machine learning',
    'dl'                     : 'deep learning',
    'cv'                     : 'computer vision',
    'ms excel'               : 'excel',
    'microsoft excel'        : 'excel',
    'python3'                : 'python',
    'reactjs'                : 'react',
    'react.js'               : 'react',
    'nodejs'                 : 'node.js',
    'tensorflow2'            : 'tensorflow',
    'sklearn'                : 'scikit-learn',
    'scikit learn'           : 'scikit-learn',
    'postgres'               : 'postgresql',
    'amazon web services'    : 'aws',
    'google cloud platform'  : 'gcp',
    'google cloud'           : 'gcp',
    'microsoft azure'        : 'azure',
    'powerbi'                : 'power bi',
    'power-bi'               : 'power bi',
    'pyspark'                : 'spark',
    'apache spark'           : 'spark',
    'github'                 : 'git',
    'git/github'             : 'git',
}


KNOWN_SKILLS = [
    # Programming
    'python', 'java', 'javascript', 'sql', 'scala',
    'ruby', 'php', 'swift', 'kotlin', 'golang',
    'matlab', 'bash', 'typescript', 'html', 'css',
    'r programming', 'c programming', 'c++',
    # Web
    'react', 'angular', 'vue', 'nodejs', 'django', 'flask',
    'fastapi', 'spring', 'jquery', 'bootstrap', 'rest api',
    'graphql', 'json', 'xml',
    # Data & ML
    'machine learning', 'deep learning', 'natural language processing',
    'computer vision', 'tensorflow', 'pytorch', 'keras',
    'scikit learn', 'pandas', 'numpy', 'matplotlib', 'opencv',
    'data analysis', 'data visualization', 'data modeling',
    'data mining', 'statistical analysis', 'predictive modeling',
    'big data', 'etl', 'data warehousing', 'data preprocessing',
    # Databases
    'mysql', 'postgresql', 'mongodb', 'redis', 'oracle',
    'sqlite', 'cassandra', 'sql server', 'dynamodb',
    # Cloud & DevOps
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
    'git', 'github', 'linux', 'unix', 'terraform', 'ansible',
    'ci cd', 'devops', 'mlops',
    # BI Tools
    'tableau', 'power bi', 'excel', 'google analytics', 'looker',
    'qlik', 'sas', 'spss',
    # Big Data
    'hadoop', 'spark', 'kafka', 'hive', 'airflow',
    'databricks', 'snowflake', 'redshift',
    # Finance Skills
    'financial modeling', 'financial analysis', 'financial reporting',
    'accounting', 'auditing', 'gaap', 'tax preparation',
    'tax planning', 'budgeting', 'forecasting', 'valuation',
    'risk management', 'quickbooks', 'sap', 'erp',
    'investment analysis', 'portfolio management',
    # HR Skills
    'recruitment', 'payroll', 'performance management',
    'employee relations', 'training development', 'onboarding',
    'hris', 'workday', 'succession planning', 'compensation',
    # Operations & Supply Chain
    'supply chain', 'procurement', 'vendor management',
    'logistics', 'inventory management', 'project management',
    'process improvement', 'lean', 'six sigma',
    # Other Tools
    'jira', 'confluence', 'salesforce', 'servicenow',
    'ms office', 'microsoft office', 'sharepoint',
    # Networking
    'networking', 'tcp ip', 'dns', 'vpn', 'firewall',
    'network security', 'cisco', 'wireless networking',
    # Security
    'cybersecurity', 'penetration testing', 'ethical hacking',
    'vulnerability assessment', 'siem', 'oauth', 'jwt',
]



# DATASET 1 — AI Job Dataset

def normalize_skill(skill):
    skill = skill.lower().strip()
    return NORMALIZATION_MAP.get(skill, skill)


def extract_from_ai_dataset():
    print("\nDataset 1: AI Job Dataset")
    print("-" * 40)

    df = pd.read_csv(AI_DATASET)
    print(f"   Rows  : {len(df):,}")
    print(f"   Roles : {df['job_title'].nunique()}")

    role_skills = defaultdict(set)

    for role, role_df in df.groupby('job_title', group_keys=False):
        for skills_str in role_df['required_skills']:
            if pd.isna(skills_str):
                continue
            skills = [normalize_skill(s.strip()) for s in skills_str.split(',')]
            skills = [s for s in skills if len(s) > 1]
            role_skills[role].update(skills)

    print(f"   Skills extracted for {len(role_skills)} roles ")
    return role_skills



# DATASET 2 — Job Descriptions Dataset

def clean_text(text):
    if pd.isna(text) or str(text).strip() == '':
        return ''

    text = str(text).lower()
    text = re.sub(r'\(e\.g\.,?\s*', '', text)
    text = re.sub(r'\(i\.e\.,?\s*', '', text)
    text = re.sub(r'[\(\)]', '', text)
    text = re.sub(r'[^a-z\s/+]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_known_skills(text):
    text = clean_text(text)
    if not text:
        return []

    found = []
    for skill in KNOWN_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text):
            found.append(skill)
    return found


def extract_from_job_desc_dataset():
    print("\nDataset 2: Job Descriptions Dataset")
    print("-" * 40)

    df = pd.read_csv(JOB_DATASET)
    print(f"   Total Rows  : {len(df):,}")
    print(f"   Total Roles : {df['Job Title'].nunique()}")

    
    df = df[df['Job Title'].isin(RELEVANT_ROLES_JOB_DESC)].copy()
    print(f"   After Filter: {len(df):,} rows | {df['Job Title'].nunique()} roles")

    
    sampled_list = []
    for role, role_df in df.groupby('Job Title', group_keys=False):
        sampled = role_df.sample(min(len(role_df), 500), random_state=42)
        sampled_list.append(sampled)
    df = pd.concat(sampled_list).reset_index(drop=True)
    print(f"   After Sample: {len(df):,} rows")

    role_skills = defaultdict(set)

    for role, role_df in df.groupby('Job Title', group_keys=False):
        for text in role_df['skills']:
            found = extract_known_skills(text)
            role_skills[role].update(found)

    print(f"   Skills extracted for {len(role_skills)} roles ")
    return role_skills



def combine_skills(ai_skills, job_desc_skills):
    print("\nCombining Both Datasets")
    print("-" * 40)

    combined = defaultdict(set)

   
    for role, skills in ai_skills.items():
        combined[role].update(skills)

    
    for role, skills in job_desc_skills.items():
        combined[role].update(skills)

    print(f"   AI Dataset roles       : {len(ai_skills)}")
    print(f"   Job Desc Dataset roles : {len(job_desc_skills)}")
    print(f"   Combined unique roles  : {len(combined)}")

    return combined




def save_combined_skills(combined_skills):
    print("\nSaving Combined Skills")
    print("-" * 40)

    rows = []
    for role, skills in sorted(combined_skills.items()):
        
        skills_list = sorted(list(skills))[:15]
        rows.append({
            'Job_Role': role,
            'Skills'  : ', '.join(skills_list)
        })

    df_output = pd.DataFrame(rows)
    df_output.to_csv(OUTPUT_FILE, index=False)

    print(f"   Saved : {OUTPUT_FILE}")
    print(f"   Total Roles : {len(df_output)}")
    print(f"\n   All Roles:")
    for _, row in df_output.iterrows():
        print(f"\n   {row['Job_Role']}:")
        print(f"      {row['Skills']}")

    return df_output




def main():
    print("COMBINED SKILL EXTRACTION")
    print("Dataset 1: AI Job Dataset")
    print("Dataset 2: Job Descriptions Dataset")

   
    ai_skills       = extract_from_ai_dataset()

    job_desc_skills = extract_from_job_desc_dataset()

    combined        = combine_skills(ai_skills, job_desc_skills)


    df_output       = save_combined_skills(combined)

    print("\nCOMBINED SKILL EXTRACTION COMPLETE")
    print(f"Output: data/skills_per_role.csv")

    return combined


if __name__ == "__main__":
    main()
