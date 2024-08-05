from fastapi import FastAPI
app = FastAPI()
from controllers.predict import router as router
app.include_router(router)