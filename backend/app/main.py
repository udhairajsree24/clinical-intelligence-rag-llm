from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.routes import (
    pubmed_routes,
    openfda_routes,
    rxnorm_routes,
    extraction_routes,
    from_note_routes,
    records_routes,
    search_routes,
    rag_routes,
    analytics_routes,
    health_routes,
)

app = FastAPI(title="Clinical Intelligence Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pubmed_routes.router)
app.include_router(openfda_routes.router)
app.include_router(rxnorm_routes.router)
app.include_router(extraction_routes.router)
app.include_router(from_note_routes.router)
app.include_router(records_routes.router)
app.include_router(search_routes.router)
app.include_router(rag_routes.router)
app.include_router(analytics_routes.router)
app.include_router(health_routes.router)


@app.get("/")
def home():
    return {"message": "Backend running"}