from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "CAT/DOG API"}

@app.get("/cats/{item_id}")
def get_cat(item_id: str, request: Request):
    user = request.headers.get('user', 'Unknown')
    return {"item_id": item_id, "user": user}

@app.get("/dogs/{item_id}")
def get_dog(item_id: str, request: Request):
    user = request.headers.get('user', 'Unknown')
    return {"item_id": item_id, "user": user}

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)