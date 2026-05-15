import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Φόρτωση
df = pd.read_csv(r"C:\PLHROFORIKH\ODEP\Sleep_health_and_lifestyle_dataset.csv")
df.columns = df.columns.str.strip()
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('No_Disorder')

# ΔΙΑΚΡΙΤΟΠΟΙΗΣΗ (Μετατροπή αριθμών σε κατηγορίες)
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 35, 50, 100], labels=['Young', 'Middle_Aged', 'Senior'])
df['Sleep_Category'] = pd.cut(df['Sleep Duration'], bins=[0, 6, 8, 24], labels=['Short_Sleep', 'Normal_Sleep', 'Long_Sleep'])
df['Stress_Category'] = pd.cut(df['Stress Level'], bins=[0, 4, 7, 10], labels=['Low_Stress', 'Med_Stress', 'High_Stress'])
df['Steps_Category'] = pd.cut(df['Daily Steps'], bins=[0, 5000, 8000, 30000], labels=['Low_Activity', 'Med_Activity', 'High_Activity'])

# Επιλογή στηλών για τον Apriori
columns_to_keep = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder', 
                   'Age_Group', 'Sleep_Category', 'Stress_Category', 'Steps_Category']
df_apriori = df[columns_to_keep]

# Μετατροπή σε One-Hot Encoding (απαραίτητο για τον Apriori στην Python)
df_encoded = pd.get_dummies(df_apriori)


# Εύρεση συχνών συνόλων αντικειμένων 
# use_colnames=True anti gia 0,1,2 o αλγόριθμος επιστρέφει τα πραγματικά ονόματα των προϊόντων (π.χ. ['Γάλα', 'Ψωμί']
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)

# Παραγωγή κανόνων βάσει Lift (Lift > 1 σημαίνει θετική συσχέτιση)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

# Ταξινόμηση κανόνων από το υψηλότερο Lift
rules = rules.sort_values('lift', ascending=False)

print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(20))

