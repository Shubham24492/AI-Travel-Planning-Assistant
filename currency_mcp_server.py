import httpx
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Currency Exchange Server")


@mcp.tool(description="Convert an amount between two currencies using current rates.")
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    from_code = from_currency.upper()
    to_code = to_currency.upper()
    response = httpx.get(
        "https://api.frankfurter.app/latest",
        params={"amount": amount, "from": from_code, "to": to_code},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "amount": amount,
        "from": from_code,
        "to": to_code,
        "converted_amount": data["rates"][to_code],
        "rate_date": data.get("date"),
    }


if __name__ == "__main__":
    mcp.run()