import urllib.request, urllib.parse, uuid
params = {'search_query': 'all:"time average neutral model" AND all:"TNTB" AND all:"model dynamics" AND au:"Danino" AND au:"Shnerb"', 'max_results': 5, 'sortBy': 'relevance'}
url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
req = urllib.request.Request(
    url,
    headers={"User-Agent": f"ResearchAssistant/1.1 (mailto:bot-{uuid.uuid4().hex[:8]}@example.com)"}
)
try:
    with urllib.request.urlopen(req, timeout=50) as r:
        print("Success!", r.status)
        print(r.read().decode("utf-8")[:200])
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Error:", e)
