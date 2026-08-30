import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.orders import settle, validate_order

app = FastAPI()

CART_SERVICE = "http://cart_service:8002"
PRODUCT_SERVICE = "http://product_service:8001"


class OrderRequest(BaseModel):
    agent_id: int
    product_id: str
    balance: float


@app.get("/health")
def health():
    return {"status": "alive"}


@app.post("/order")
async def place_order(req: OrderRequest):
    """Fetch the order's inputs, apply core.orders' rules, then commit.

    The rules themselves are deliberately not here: live/engine.py applies the
    same validate_order to the same shapes, so a purchase the service would
    refuse cannot quietly succeed in the simulation.
    """
    async with httpx.AsyncClient() as client:
        cart = (await client.get(f"{CART_SERVICE}/cart/{req.agent_id}")).json()

        prod_resp = await client.get(f"{PRODUCT_SERVICE}/products/{req.product_id}")
        product = None if prod_resp.status_code == 404 else prod_resp.json()

        rejection = validate_order(cart, product, req.balance)
        if rejection is not None:
            raise HTTPException(status_code=400, detail=rejection.value)

        dec_resp = await client.post(
            f"{PRODUCT_SERVICE}/products/{req.product_id}/decrement_stock"
        )
        if dec_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to decrement stock")

        await client.delete(f"{CART_SERVICE}/cart/{req.agent_id}")

    return {"success": True, "new_balance": settle(req.balance, product["price"])}
