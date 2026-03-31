import pandas as pd
import os
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score,
                              confusion_matrix, classification_report)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
INPUT_FILE  = os.path.join(DATA_DIR, 'cleaned_student_data.csv')

os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    'CGPA',
    'Internships',
    'Projects',
    'Coding_Skills',
    'Communication_Skills',
    'Aptitude_Test_Score',
    'Soft_Skills_Rating',
    'Certifications',
    'Backlogs'
]
TARGET_COL = 'Placement_Label'


def step1_load_data():
    print("\nSTEP 1: Loading Cleaned Student Data")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Rows       : {df.shape[0]:,}")
    print(f"   Columns    : {df.shape[1]}")
    print(f"   Placed     : {int(df[TARGET_COL].sum()):,}")
    print(f"   Not Placed : {int((df[TARGET_COL]==0).sum()):,}")
    return df


def step2_prepare_features(df):
    print("\nSTEP 2: Preparing Features and Target")
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    print(f"   Features : {FEATURE_COLS}")
    print(f"   X Shape  : {X.shape}")
    print(f"   y Shape  : {y.shape}")
    return X, y


def step3_train_test_split(X, y):
    print("\nSTEP 3: Train Test Split (80-20)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"   Training Set : {X_train.shape[0]:,} rows")
    print(f"   Testing Set  : {X_test.shape[0]:,} rows")
    return X_train, X_test, y_train, y_test


def step4_train_xgboost(X_train, y_train):
    print("\nSTEP 4: Training XGBoost Model")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    print(f"   XGBoost trained successfully")
    print(f"   n_estimators : 200")
    print(f"   max_depth    : 6")
    print(f"   learning_rate: 0.1")
    return model


def step5_evaluate_model(model, X_train, y_train, X_test, y_test):
    print("\nSTEP 5: Evaluating Model")


    train_pred     = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)

    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, y_pred)
    precision     = precision_score(y_test, y_pred)
    recall        = recall_score(y_test, y_pred)
    f1            = f1_score(y_test, y_pred)
    auc           = roc_auc_score(y_test, y_prob)

    print(f"\n   Training Accuracy : {train_accuracy*100:.2f}%")
    print(f"   Test Accuracy     : {test_accuracy*100:.2f}%")
    print(f"   Precision         : {precision*100:.2f}%")
    print(f"   Recall            : {recall*100:.2f}%")
    print(f"   F1 Score          : {f1*100:.2f}%")
    print(f"   ROC-AUC           : {auc:.3f}")

    diff = train_accuracy - test_accuracy
    if diff > 0.05:
        print(f"\n   Warning: Overfitting detected! Difference = {diff*100:.2f}%")
    else:
        print(f"\n   No Overfitting detected")

    print(f"\n   Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"   FN={cm[1][0]}  TP={cm[1][1]}")

    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['Not Placed', 'Placed']))

    return {
        'Training Accuracy': round(train_accuracy*100, 2),
        'Test Accuracy'    : round(test_accuracy*100, 2),
        'Precision'        : round(precision*100, 2),
        'Recall'           : round(recall*100, 2),
        'F1 Score'         : round(f1*100, 2),
        'ROC-AUC'          : round(auc, 3)
    }


def step6_feature_importance(model):
    print("\nSTEP 6: Feature Importance")
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        'Feature'   : FEATURE_COLS,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    print(f"\n   Feature Importances:")
    for _, row in feat_imp.iterrows():
        bar = '#' * int(row['Importance'] * 50)
        print(f"   {row['Feature']:30} {row['Importance']:.4f}  {bar}")

    feat_imp.to_csv(os.path.join(MODELS_DIR, 'feature_importance.csv'), index=False)
    print(f"\n   Saved: models/feature_importance.csv")
    return feat_imp


def step7_cross_validation(model, X, y):
    print("\nSTEP 7: Cross Validation (5-Fold)")
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"   Fold Scores : {[round(s*100,2) for s in scores]}")
    print(f"   Mean        : {scores.mean()*100:.2f}%")
    print(f"   Std         : {scores.std()*100:.2f}%")
    if scores.std() < 0.02:
        print(f"   No Overfitting detected")
    else:
        print(f"   Warning: Possible Overfitting")


def step8_save_model(model):
    print("\nSTEP 8: Saving Model")
    model_path = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"   Saved: {model_path}")


def calculate_readiness_score(student):
    score = 0

    
    score += (student['CGPA'] / 10) * 30

    score += (student['Coding_Skills'] / 10) * 20

   
    score += (student['Communication_Skills'] / 10) * 15


    score += (student['Aptitude_Test_Score'] / 100) * 15


    score += min(student['Internships'] / 3, 1) * 10


    score += min(student['Projects'] / 5, 1) * 5

   
    score += min(student['Certifications'] / 3, 1) * 5

   
    score -= (student['Backlogs'] / 3) * 10

    return round(score, 1)


def get_readiness_label(score):
    if score >= 70:
        return "Highly Ready"
    elif score >= 50:
        return "Moderately Ready"
    else:
        return "Needs Improvement"


def step9_predict_sample(model):
    print("\nSTEP 9: Sample Prediction")

    sample_student = {
        'CGPA'                : 6.0,
        'Internships'         : 1,
        'Projects'            : 2,
        'Coding_Skills'       : 5,
        'Communication_Skills': 5,
        'Aptitude_Test_Score' : 60,
        'Soft_Skills_Rating'  : 6,
        'Certifications'      : 1,
        'Backlogs'            : 1
    }

    input_df = pd.DataFrame([sample_student])

    prediction    = model.predict(input_df)[0]
    result        = "Placed" if prediction == 1 else "Not Placed"

    
    score         = calculate_readiness_score(sample_student)
    label         = get_readiness_label(score)

    print(f"\n   Sample Student:")
    for k, v in sample_student.items():
        print(f"      {k:30} : {v}")

    print(f"\n   Placement Prediction  : {result}")
    print(f"   Readiness Score       : {score}/100")
    print(f"   Readiness Label       : {label}")

    print(f"\n   Score Breakdown:")
    print(f"      CGPA contribution          : {round((sample_student['CGPA']/10)*30, 1)} / 30")
    print(f"      Coding contribution        : {round((sample_student['Coding_Skills']/10)*20, 1)} / 20")
    print(f"      Communication contribution : {round((sample_student['Communication_Skills']/10)*15, 1)} / 15")
    print(f"      Aptitude contribution      : {round((sample_student['Aptitude_Test_Score']/100)*15, 1)} / 15")
    print(f"      Internships contribution   : {round(min(sample_student['Internships']/3,1)*10, 1)} / 10")
    print(f"      Projects contribution      : {round(min(sample_student['Projects']/5,1)*5, 1)} / 5")
    print(f"      Certifications contribution: {round(min(sample_student['Certifications']/3,1)*5, 1)} / 5")
    print(f"      Backlogs penalty           : -{round((sample_student['Backlogs']/3)*10, 1)} / 10")


def main():
    print("MODEL TRAINING — XGBoost")

    df                               = step1_load_data()
    X, y                             = step2_prepare_features(df)
    X_train, X_test, y_train, y_test = step3_train_test_split(X, y)
    model                            = step4_train_xgboost(X_train, y_train)
    metrics                          = step5_evaluate_model(model, X_train, y_train, X_test, y_test)
    step6_feature_importance(model)
    step7_cross_validation(model, X, y)
    step8_save_model(model)
    step9_predict_sample(model)

    print("\nMODEL TRAINING COMPLETE")
    print(f"Model saved      : models/xgboost_model.pkl")
    print(f"Train Accuracy   : {metrics['Training Accuracy']}%")
    print(f"Test Accuracy    : {metrics['Test Accuracy']}%")
    print(f"F1 Score         : {metrics['F1 Score']}%")

    return model


if __name__ == "__main__":
    main()
