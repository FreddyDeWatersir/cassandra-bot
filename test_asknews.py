"""
AskNews diagnostic script - tests both auth methods and prints detailed info.
Run with: poetry run python test_asknews.py
"""
import os
import asyncio
import dotenv
dotenv.load_dotenv()

async def test_asknews():
    print("=" * 60)
    print("ASKNEWS DIAGNOSTIC")
    print("=" * 60)

    # Check what env vars are set
    client_id = os.getenv("ASKNEWS_CLIENT_ID")
    client_secret = os.getenv("ASKNEWS_SECRET")
    api_key = os.getenv("ASKNEWS_API_KEY")

    print(f"\nASKNEWS_CLIENT_ID: {'SET (' + client_id[:8] + '...)' if client_id else 'NOT SET'}")
    print(f"ASKNEWS_SECRET:    {'SET (' + client_secret[:8] + '...)' if client_secret else 'NOT SET'}")
    print(f"ASKNEWS_API_KEY:   {'SET (' + api_key[:8] + '...)' if api_key else 'NOT SET'}")

    # Test 1: Try with API key directly using the SDK
    if api_key:
        print(f"\n{'=' * 60}")
        print("TEST 1: Direct SDK call with API key")
        print("=" * 60)
        try:
            from asknews_sdk import AsyncAskNewsSDK
            async with AsyncAskNewsSDK(
                api_key=api_key,
                scopes=["news"],
            ) as ask:
                response = await ask.news.search_news(
                    query="latest technology news",
                    n_articles=3,
                    return_type="string",
                    method="kw",
                )
                print(f"SUCCESS! Got {len(response.as_string)} chars")
                print(f"Preview: {response.as_string[:200]}...")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")

    # Test 2: Try with OAuth2 directly using the SDK
    if client_id and client_secret:
        print(f"\n{'=' * 60}")
        print("TEST 2: Direct SDK call with OAuth2 (client_id + secret)")
        print("=" * 60)
        try:
            from asknews_sdk import AsyncAskNewsSDK
            async with AsyncAskNewsSDK(
                client_id=client_id,
                client_secret=client_secret,
                scopes=["news"],
            ) as ask:
                response = await ask.news.search_news(
                    query="latest technology news",
                    n_articles=3,
                    return_type="string",
                    method="kw",
                )
                print(f"SUCCESS! Got {len(response.as_string)} chars")
                print(f"Preview: {response.as_string[:200]}...")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")

    # Test 3: Try with forecasting-tools AskNewsSearcher
    print(f"\n{'=' * 60}")
    print("TEST 3: forecasting-tools AskNewsSearcher wrapper")
    print("=" * 60)
    try:
        from forecasting_tools import AskNewsSearcher
        searcher = AskNewsSearcher()
        result = await searcher.call_preconfigured_version(
            "asknews/news-summaries",
            "What are the latest developments in artificial intelligence?"
        )
        print(f"SUCCESS! Got {len(result)} chars")
        print(f"Preview: {result[:200]}...")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")

    # Test 4: Try API key with different scopes
    if api_key:
        print(f"\n{'=' * 60}")
        print("TEST 4: API key with no scopes specified")
        print("=" * 60)
        try:
            from asknews_sdk import AsyncAskNewsSDK
            async with AsyncAskNewsSDK(
                api_key=api_key,
            ) as ask:
                response = await ask.news.search_news(
                    query="latest technology news",
                    n_articles=3,
                    return_type="string",
                    method="kw",
                )
                print(f"SUCCESS! Got {len(response.as_string)} chars")
                print(f"Preview: {response.as_string[:200]}...")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")

    # Test 5: Try the profile endpoint to see account info
    if api_key:
        print(f"\n{'=' * 60}")
        print("TEST 5: Check account profile via API")
        print("=" * 60)
        try:
            import httpx
            headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.asknews.app/v1/profiles/me",
                    headers=headers,
                )
                print(f"Status: {resp.status_code}")
                print(f"Response: {resp.text[:500]}")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")

    if client_id and client_secret:
        print(f"\n{'=' * 60}")
        print("TEST 6: Check account profile via OAuth2")
        print("=" * 60)
        try:
            import httpx
            # First get OAuth2 token
            async with httpx.AsyncClient() as client:
                token_resp = await client.post(
                    "https://auth.asknews.app/oauth2/token",
                    data={
                        "grant_type": "client_credentials",
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "scope": "news",
                    },
                )
                print(f"Token response status: {token_resp.status_code}")
                if token_resp.status_code == 200:
                    token = token_resp.json().get("access_token", "")
                    headers = {"Authorization": f"Bearer {token}"}
                    resp = await client.get(
                        "https://api.asknews.app/v1/profiles/me",
                        headers=headers,
                    )
                    print(f"Profile status: {resp.status_code}")
                    print(f"Profile response: {resp.text[:500]}")
                else:
                    print(f"Token response: {token_resp.text[:500]}")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")

    print(f"\n{'=' * 60}")
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_asknews())