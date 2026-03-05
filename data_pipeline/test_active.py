import asyncio
import httpx
from data_pipeline.live_orderbook_recorder import get_active_btc_slugs

async def main():
    slugs = await get_active_btc_slugs()
    print("Generated Slugs:", slugs)
    async with httpx.AsyncClient() as client:
        for slug in slugs:
            url = f"https://gamma-api.polymarket.com/events?slug={slug}"
            res = await client.get(url)
            print(f"Slug: {slug} | Status: {res.status_code} | HasData: {bool(res.json())}")

asyncio.run(main())
