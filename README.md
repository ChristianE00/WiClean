
# Running with Docker

#### Note: Docker should only use small queries as large queries will crash the kernel.

1. Open Docker
2. cd into the project directory and run the following command to lauch the container:
```bash
docker-compose up --build
```

3. Now, simpy running all code blocks will run and display the model results.



# Running without Docker


#### Install packages
If you wish to run without docker, you will need to install the following packages:

```bash
pip install jupyter pandas scikit-learn SPARQLWrapper 
```

#### Changing the query size

If you wish to run with a different query size, simply change the value after 'LIMIT' on line 36 to your choosing.

#### Running the Code

To run the code, enter the following command in the terminal:
```bash
py .\graph.py
```

