# 20220301.frr --Smallest dataset for testing

import pandas as pd
import re
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("wikipedia", "20220301.frr", trust_remote_code=True)
print('dataset loaded')

df = pd.DataFrame(dataset['train'])

# Preprocessing steps
columns_to_keep = ['id', 'url', 'title', 'text']
df = df[columns_to_keep]

# Handle missing values
df['text'] = df['text'].fillna('')

# Standardize entity names
def standardize_entity_name(title):
    # Implement your entity name standardization logic here
    # For example, you can use regular expressions or string manipulation
    standardized_title = title.lower().replace(' ', '_')
    return standardized_title

df['title'] = df['title'].apply(standardize_entity_name)

# Function to extract entity mentions from text
def extract_entity_mentions(text, graph):
    entity_mentions = []
    
    # Iterate over the nodes in the graph
    for entity_id, node in graph.items():
        entity_title = node['title']
        
        # Check if the entity title is mentioned in the text
        if re.search(r'\b' + re.escape(entity_title) + r'\b', text, re.IGNORECASE):
            entity_mentions.append(entity_id)
    
    return entity_mentions

graph = {}

# Iterate over the DataFrame rows
for _, row in df.iterrows():
    entity_id = row['id']
    entity_title = row['title']
    entity_url = row['url']
    entity_text = row['text']
    
    node = {
        'id': entity_id,
        'title': entity_title,
        'url': entity_url,
        'text': entity_text,
        'relationships': []
    }
    
    graph[entity_id] = node

# Create interlinks between nodes
for entity_id, node in graph.items():
    entity_text = node['text']
    
    entity_mentions = extract_entity_mentions(entity_text, graph)
    
    # Create relationships with the mentioned entities
    for mentioned_entity_id in entity_mentions:
        if mentioned_entity_id != entity_id:
            relationship = {
                'target': mentioned_entity_id,
                'type': 'mentions'
            }
            node['relationships'].append(relationship)

print(f"Number of nodes in the graph: {len(graph)}")
total_relationships = sum(len(node['relationships']) for node in graph.values())
print(f"Total number of relationships in the graph: {total_relationships}")