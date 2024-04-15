# 20220301.frr --Smallest dataset for testing
import pandas as pd
import re
import sys
import os
import json
from datasets import load_dataset


COUNTER = 0

class Graph:
    def __init__(self):
        self.graph = {}
        self.relationships = []
        self.dataset = None

        if os.path.exists('./graph.json'):
            print('graph exists')
            with open('graph.json') as g:
                self.graph = json.load(g)
            print(f"Number of nodes in the graph: {len(self.graph)}")
            total_relationships = sum(len(node['relationships']) for node in self.graph.values())
            print(f"Total number of relationships in the graph: {total_relationships}")
        else:
            print('graph does NOT exist')

    def train_small_graph(self):
        '''
        Load "20220301.frr"
        Size: 15,199 rows
        '''
        self.dataset = load_dataset("wikipedia", "20220301.frr", trust_remote_code=True)
        print('dataset loaded')
        return 
    
    def train_medium_graph(self):
        '''
        Load "20220301.simple"
        Size: 205,328 rows
        '''
        self.dataset = load_dataset("wikipedia", "20220301.frr", trust_remote_code=True)
        print('dataset loaded')
        return 

    def train_large_graph(self):
        '''
        Load "20220301.en"
        Size: 6,458,670 rows
        '''
        self.dataset = load_dataset("wikipedia", "20220301.frr", trust_remote_code=True)
        print('dataset loaded')
        return

    def train(self):
        # Preprocessing steps
        df = pd.DataFrame(self.dataset['train'])
        columns_to_keep = ['id', 'url', 'title', 'text']
        df = df[columns_to_keep]

        # Handle missing values & standardize entity names
        df['text'] = df['text'].fillna('')
        df['title'] = df['title'].apply(self.standardize_entity_name)

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
            self.graph[entity_id] = node

        # Create interlinks between nodes
        print(' Graph size: ', len(self.graph))
        for entity_id, node in self.graph.items():
            '''
            if COUNTER > 10:
                print('ENDING')
                self.save_graph()
                return
            '''
            entity_text = node['text']
            entity_mentions = self.extract_entity_mentions(entity_text, self.graph)
            # Create relationships with the mentioned entities
            for mentioned_entity_id in entity_mentions:
                    if mentioned_entity_id != entity_id:
                        relationship = {
                            'target': mentioned_entity_id,
                            'type': 'mentions'
                        }
                        node['relationships'].append(relationship)

        print(f"Number of nodes in the graph: {len(self.graph)}")
        total_relationships = sum(len(node['relationships']) for node in self.graph.values())
        print(f"Total number of relationships in the graph: {total_relationships}")

    def save_graph(self):
        print('saving graph...')
        with open('graph.json', 'w') as file:
            json.dump(self.graph, file)
        print('graph saved successfully')


    def standardize_entity_name(self, title):
        '''Standardize entity names
        '''

        # Implement your entity name standardization logic here
        # For example, you can use regular expressions or string manipulation
        standardized_title = title.lower().replace(' ', '_')
        return standardized_title

    def extract_entity_mentions(self, text, graph):
        '''Function to extract entity mentions from text
        '''

        global COUNTER
        COUNTER += 1
        print('entered extract_entity_mentions(): ', COUNTER)
        entity_mentions = []
        
        # Iterate over the nodes in the graph
        for entity_id, node in graph.items():
            entity_title = node['title']
            # Check if the entity title is mentioned in the text
            if re.search(r'\b' + re.escape(entity_title) + r'\b', text, re.IGNORECASE):
                entity_mentions.append(entity_id)
        
        return entity_mentions


if __name__ == '__main__':
    graph = Graph()    
    input = sys.argv[1]
    if len(graph.graph) == 0:
        print('Creating new graph')
        if input == '--medium':
            graph.train_medium_graph()
        elif input == '--large':
            graph.train_large_graph()
        else:
            graph.train_small_graph()
        graph.train()
    else:
        print('graph loaded from existing file')


