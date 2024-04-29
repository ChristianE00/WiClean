from SPARQLWrapper import SPARQLWrapper, JSON
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import datetime

def save_graph(results):
    print('Saving graph...')
    with open('graph.json', 'w') as file:
        json.dump(results, file)
    print('Graph saved successfully')

def add_inconsistencies(data):
    # Add inconsistencies
    for row in data:
        if random.random() < 0.5:  # 50% chance to add an inconsistency
            date_of_death = datetime.datetime.now() - datetime.timedelta(days=random.randint(91, 365))
            row[2] = date_of_death.strftime("%Y-%m-%d")  # Update the date format to match the original data
            row.append(1)  # Add a label indicating an inconsistency
        else:
            row.append(0)  # Add a label indicating consistency

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

# Add artificial inconsistencies
add_inconsistencies(data)

# Create a DataFrame
df = pd.DataFrame(data, columns=["person", "spouse", "date_of_death", "inconsistent"])

# Feature engineering
df["date_of_death"] = pd.to_datetime(df["date_of_death"], format="%Y-%m-%d", errors="coerce")
df["days_since_death"] = (pd.Timestamp.now() - df["date_of_death"]).dt.days

# Handle missing or invalid dates
df["date_of_death"] = df["date_of_death"].fillna(pd.NaT)
df["days_since_death"] = df["days_since_death"].fillna(0)

# Split the data
X = df[["days_since_death"]]
y = df["inconsistent"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the class distribution
if len(y_train.unique()) == 1:
    majority_class = y_train.unique()[0]
    print(f"All instances belong to class {majority_class}. Skipping classification.")
    y_pred = [majority_class] * len(y_test)
else:
    # Train and evaluate different classifiers
    classifiers = [
        ('Decision Tree', DecisionTreeClassifier()),
        ('Random Forest', RandomForestClassifier()),
        ('SVM', SVC())
    ]

    for name, clf in classifiers:
        print(f"\nClassifier: {name}")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)