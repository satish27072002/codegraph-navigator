from fastapi import FastAPI

app = FastAPI(title="ingest_service")


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}
