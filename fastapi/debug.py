import uvicorn
import os

# Must be set before importing main
os.environ["FASTAPI_ENV"] = "development"
import api.main

"""Run your FastAPI application in debug mode in VS Code.
"""

if __name__ == "__main__":
    uvicorn.run(api.main.app, host="127.0.0.1", port=80)