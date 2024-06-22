def query(payload):
    """
    Sends a payload to a TGI endpoint.
    """
    API_URL = "http://nexusraven.nexusflow.ai"
    headers = {
        "Content-Type": "application/json"
    }
    import requests
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def query_raven(prompt):
	"""
	This function sends a request to the TGI endpoint to get Raven's function call.
	This will not generate Raven's justification and reasoning for the call, to save on latency.
	"""
	import requests
	output = query({
		"inputs": prompt,
		"parameters" : {"temperature" : 0.001, "stop" : ["<bot_end>"], "do_sample" : False, "max_new_tokens" : 2048, "return_full_text" : False}})
	call = output[0]["generated_text"].replace("Call:", "").strip()
	return call

def query_raven_with_reasoning(prompt):
	"""
	This function sends a request to the TGI endpoint to get Raven's function call AND justification for the call
	"""
	import requests
	output = query({
		"inputs": prompt,
		"parameters" : {"temperature" : 0.001, "do_sample" : False, "max_new_tokens" : 2000}})
	call = output[0]["generated_text"].replace("Call:", "").strip()
	return call

def execute_sql(sql_code : str):
    import sqlite3
    
    # Connect to the database
    conn = sqlite3.connect('toy_database.db')
    cursor = conn.cursor()
    
    cursor.execute('PRAGMA table_info(toys)')
    columns = [info[1] for info in cursor.fetchall()]  # Extracting the column names
    
    # Query to select all data
    cursor.execute(sql_code)
    rows = cursor.fetchall()
    
    return_string = " ".join(columns)
    for idx, row in enumerate(rows):
        row = (idx, *row)
        return_string += "\n" + str(row)
    
    # Close the connection
    conn.close()
    return return_string

def create_random_database():
    import sqlite3
    import random

    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('toy_database.db')
    
    # Create a cursor object using the cursor() method
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''CREATE TABLE IF NOT EXISTS toys
                   (id INTEGER PRIMARY KEY, name TEXT, price REAL)''')
    
    # Define some random prefixes and suffixes for toy names
    prefixes = ['Magic', 'Super', 'Wonder', 'Mighty', 'Happy', 'Crazy']
    suffixes = ['Bear', 'Car', 'Doll', 'Train', 'Dragon', 'Robot']
    
    # Insert 100 sample data rows with random names
    for i in range(1, 101):
        toy_name = random.choice(prefixes) + ' ' + random.choice(suffixes)
        toy_price = round(random.uniform(5, 20), 2)  # Random price between 5 and 20
        cursor.execute('INSERT INTO toys (name, price) VALUES (?, ?)', (toy_name, toy_price))
    
    # Commit the transaction
    conn.commit()
    
    # Query the database
    cursor.execute('SELECT * FROM toys')
    print("Toys in database:")
    for row in cursor.fetchall():
        print(row)
    
    # Close the connection
    conn.close()