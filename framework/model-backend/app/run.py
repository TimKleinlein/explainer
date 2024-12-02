import uvicorn
import os
from app import start_app

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #NOTYET

if __name__ == "__main__":
    app = start_app()
    uvicorn.run(app, host="0.0.0.0", port=5000)