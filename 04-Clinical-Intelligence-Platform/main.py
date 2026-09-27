from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import chatbot, heart_disease

app = FastAPI(title="Clinical Intelligence Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chatbot.router, prefix="/chat")
app.include_router(heart_disease.router, prefix="/heart")

##