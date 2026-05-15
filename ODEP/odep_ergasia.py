import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from imblearn.over_sampling import SMOTE

# --- 1. ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ ---
file_path = r"C:\PLHROFORIKH\ODEP\Sleep_health_and_lifestyle_dataset.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

print("✅ Το αρχείο φορτώθηκε!")

# --- 2. ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ ---

# α. Διαχείριση NaN στη στήλη Sleep Disorder
# Τα NaN σημαίνουν "Καμία Διαταραχή", οπότε τα ονομάζουμε 'None'
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

# β. Διαχωρισμός Blood Pressure
if 'Blood Pressure' in df.columns:
    df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df.drop('Blood Pressure', axis=1, inplace=True)

# γ. Encoding (Μετατροπή κειμένου σε αριθμούς)
le = LabelEncoder() 
df['Gender'] = le.fit_transform(df['Gender'])
df['BMI Category'] = le.fit_transform(df['BMI Category'])
df = pd.get_dummies(df, columns=['Occupation'])

# δ. Προετοιμασία X και y
X = df.drop(['Person ID', 'Sleep Disorder'], axis=1)
y = df['Sleep Disorder']

# ε. Scaling & SMOTE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# --- 3. ΕΚΠΑΙΔΕΥΣΗ (Διορθωμένο AdaBoost) ---
models = {
    "Decision Tree (C4.5)": DecisionTreeClassifier(criterion='entropy', random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42) # Αφαιρέθηκε το 'algorithm'
}

results = {}
print("\n--- ΑΠΟΤΕΛΕΣΜΑΤΑ ---")

for name, model in models.items():
    model.fit(X_train, y_train)  # Το 'model' περιέχει τον αλγόριθμο  Το 'name' περιέχει το όνομα (π.χ. "kNN")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n⭐ {name} | Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))

# --- 4. ΓΡΑΦΗΜΑ ---
plt.figure(figsize=(10, 6))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette='viridis')
plt.title('Σύγκριση Ακρίβειας Αλγορίθμων')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)
plt.show()

import matplotlib.pyplot as plt

# Παράδειγμα δεδομένων (αντικατέστησε με τις πραγματικές τιμές σου)
models = ['Decision Tree', 'kNN', 'Naive Bayes', 'Random Forest', 'AdaBoost']
accuracy_scores = [0.85, 0.88, 0.79, 0.89, 0.94]  # Υποθετικές τιμές

plt.figure(figsize=(10, 6))

# Δημιουργία του bar chart
bars = plt.bar(models, accuracy_scores, color=['#A8DADC', '#457B9D', '#1D3557', '#E63946', '#D62828'])

# Προσθήκη τίτλων και ετικετών
plt.title('Σύγκριση Ακρίβειας Μοντέλων (Zoom στην περιοχή 0.7 - 1.0)', fontsize=14)
plt.ylabel('Accuracy Score', fontsize=12)
plt.xlabel('Αλγόριθμοι', fontsize=12)

# --- ΤΟ ΖΟΥΜ ---
# Ορίζουμε το κάτω όριο στο 0.7 για να φαίνονται καθαρά οι διαφορές στην κορυφή
plt.ylim(0.7, 1.0) 

# Προσθήκη των τιμών πάνω από κάθε μπάρα για απόλυτη σαφήνεια
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.3f}', 
             ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()