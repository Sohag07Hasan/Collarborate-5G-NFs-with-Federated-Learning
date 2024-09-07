Clients:
	- 4 Clients (c1.py, c2.py, c3.py, c4.py)
Server:
	- 1 Server (server.py)

Datasets:
	- Whole datasets: ./Datasets/client_{client_non}/dataset.csv (not in use in ML)
	- Training/Test datasets: ./Datasets/client_{client_non}/train/client_{client_non}_train/test.csv (in use in ML)

Configuration:
	- config.json holds all the configuration of the Federated Learning Technique
	- configurations can be updated based on necessity

Run the Code: 
	- run server.py
	- run c1.py
	- run c2.py
	- run c3.py
	- run c4.py
Results:
	- After training global model will be saved in ./saved_models
	- run inference.ipynb to get the results in notebook


	
		