from fastapi import FastAPI

app = FastAPI(title="retrieval_service")


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}
