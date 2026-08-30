from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.catalog import load_catalog

app = FastAPI()

DATA_FILE = "data/raw/All_Beauty.jsonl"


class Product(BaseModel):
    """The wire contract. Catalog construction and pricing live in core/."""

    product_id: str
    name: str
    category: str
    price: float
    stock: int


_catalog: list[Product] = [Product(**row) for row in load_catalog(DATA_FILE)]
_index: dict[str, Product] = {p.product_id: p for p in _catalog}


@app.get("/health")
def health():
    return {"status": "alive", "catalog_size": len(_catalog)}


@app.get("/products", response_model=list[Product])
def get_products():
    return _catalog


@app.get("/products/category/{category}", response_model=list[Product])
def get_by_category(category: str):
    return [p for p in _catalog if p.category == category]


@app.get("/products/{product_id}", response_model=Product)
def get_product(product_id: str):
    p = _index.get(product_id)
    if not p:
        raise HTTPException(status_code=404, detail="Product not found")
    return p


@app.post("/products/{product_id}/decrement_stock")
def decrement_stock(product_id: str):
    p = _index.get(product_id)
    if not p:
        raise HTTPException(status_code=404, detail="Product not found")
    if p.stock <= 0:
        raise HTTPException(status_code=400, detail="Out of stock")
    p.stock -= 1
    return {"product_id": product_id, "stock": p.stock}
