from SPARQLWrapper import SPARQLWrapper, JSON
import json
import os
def save_graph(results):
    print('saving graph...')
    with open('graph.json', 'w') as file:
        json.dump(results, file)
    print('graph saved successfully')

endpoint_url = "https://query.wikidata.org/sparql"

sparql = SPARQLWrapper(endpoint_url)

query = """
    SELECT ?person ?spouse ?dateOfDeath WHERE {
        ?person wdt:P26 ?spouse .  # P26 represents the "spouse" property
        ?spouse wdt:P570 ?dateOfDeath .  # P570 represents the "date of death" property
        FILTER (?dateOfDeath < NOW())
    }
    LIMIT 100
            """

sparql.setQuery(query)
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

#print(results)
save_graph(results)

'''
for result in results["results"]["bindings"]:
    item_uri = result["item"]["value"]
    item_label = result["itemLabel"]["value"]
    print(f"Item URI: {item_uri}")
    print(f"Item Label: {item_label}")
    print("---")
'''
