from fastapi import FastAPI

app = FastAPI(title="workers")


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}
