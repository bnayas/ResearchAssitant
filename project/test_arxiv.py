import urllib.request, urllib.parse
params = {'search_query': 'all:"time average neutral model" AND all:"TNTB" AND all:"model dynamics" AND au:"Danino" AND au:"Shnerb"', 'max_results': 5, 'sortBy': 'relevance'}
url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
req = urllib.request.Request(
    url,
    headers={"User-Agent": "ResearchAssistant/1.0 (mailto:admin@example.com)"}
)
try:
    with urllib.request.urlopen(req, timeout=50) as r:
        print("Success!", r.status)
        print(r.read().decode("utf-8")[:200])
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Error:", e)
    if hasattr(e, 'code'):
        print("Code:", e.code)
    if hasattr(e, 'headers'):
        print("Headers:", e.headers)
