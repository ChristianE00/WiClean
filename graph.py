from SPARQLWrapper import SPARQLWrapper, JSON
import json
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
import random
import datetime

def save_graph(results):
    print('Saving graph...')
    with open('graph.json', 'w') as file:
        json.dump(results, file)
    print('Graph saved successfully')

def add_inconsistencies(data):
    # Add inconsistencies randomly
    for row in data:
        if random.random() < 0.5:  # 50% chance to add an inconsistency
            row.append(1)  # Inconsistent
        else:
            row.append(0)  # Consistent

endpoint_url = "https://query.wikidata.org/sparql"
sparql = SPARQLWrapper(endpoint_url)

query = """
    SELECT ?person ?spouse ?dateOfDeath WHERE {
        ?person wdt:P26 ?spouse .  # P26 represents the "spouse" property
        ?spouse wdt:P570 ?dateOfDeath .  # P570 represents the "date of death" property
        FILTER (?dateOfDeath < NOW())
    }
    LIMIT 5000
"""

sparql.setQuery(query)
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

# Preprocess the query results
data = []
for result in results["results"]["bindings"]:
    person = result["person"]["value"]
    spouse = result["spouse"]["value"]
    date_of_death_uri = result["dateOfDeath"]["value"]
    date_of_death = date_of_death_uri.split("/")[-1]  # Extract the date from the URI
    data.append([person, spouse, date_of_death])

# Add inconsistencies randomly
add_inconsistencies(data)

# Create a DataFrame
df = pd.DataFrame(data, columns=["person", "spouse", "date_of_death", "inconsistent"])

# Feature engineering
df["person"] = df["person"].apply(lambda x: x.split("/")[-1])  # Extract the person ID
df["spouse"] = df["spouse"].apply(lambda x: x.split("/")[-1])  # Extract the spouse ID
df["date_of_death"] = pd.to_datetime(df["date_of_death"], format="%Y-%m-%d", errors="coerce")
df["days_since_death"] = (pd.Timestamp.now() - df["date_of_death"]).dt.days

# Handle missing or invalid dates
df["date_of_death"] = df["date_of_death"].fillna(pd.NaT)
df["days_since_death"] = df["days_since_death"].fillna(0)

# One-hot encode categorical features
onehot_encoder = OneHotEncoder()
encoded_features = onehot_encoder.fit_transform(df[["person", "spouse"]])
encoded_feature_names = onehot_encoder.get_feature_names_out(["person", "spouse"])
encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoded_feature_names)

# Concatenate encoded features with numerical features
X = pd.concat([encoded_df, df["days_since_death"]], axis=1)
y = df["inconsistent"]

# Train and evaluate classifiers using cross-validation
classifiers = [
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('SVM', SVC())
]

for name, clf in classifiers:
    print(f"\nClassifier: {name}")
    scores = cross_val_score(clf, X, y, cv=5)
    print("Cross-validation scores:", scores)
    print("Mean score:", scores.mean())

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the best classifier on the test set
best_clf = RandomForestClassifier()
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nBest Classifier: Random Forest")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)