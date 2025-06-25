from colorama import Fore
from mcp.server.fastmcp import FastMCP
import json 
import requests

mcp = FastMCP("doctorserver")

# Build server function
@mcp.tool()
def list_doctors(state:str) -> str:
    """This tool returns doctors that may be near you.
    Args:
        state: the two letter state code that you live in. 
        Example payload: "CA"

    Returns:
        str: a list of doctors that may be near you
        Example Response "{"DOC001":{"name":"Dr John James", "specialty":"Cardiology"...}...}" 
        """

    url = 'https://raw.githubusercontent.com/nicknochnack/ACPWalkthrough/refs/heads/main/doctors.json'
    resp = requests.get(url)
    doctors = json.loads(resp.text)

    matches = [doctor for doctor in doctors.values() if doctor['address']['state'] == state]    
    return str(matches) 

# Kick off server if file is run 
if __name__ == "__main__":
    mcp.run(transport="stdio")
