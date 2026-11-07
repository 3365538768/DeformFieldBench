"""Read public Kaggle metadata; never read local account credentials."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]/"data/metadata"
BASE="https://www.kaggle.com/api/v1/datasets/"
SLUG="anonymous336/physical-field-material-parameter-5000"


def fetch(item):
    name,url=item
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"FieldBridge/0.1"}),timeout=45) as r:
            data=r.read(8000000)
            parsed=json.loads(data)
        (ROOT/(name+".json")).write_text(json.dumps(parsed,indent=2))
        print(json.dumps({"name":name,"keys":list(parsed) if isinstance(parsed,dict) else "list","bytes":len(data)}))
    except Exception as e:
        print(json.dumps({"name":name,"error":repr(e)}))


if __name__=="__main__":
    ROOT.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(fetch,[("kaggle_view",BASE+"view/"+SLUG),("kaggle_files",BASE+"list/"+SLUG+"?pageSize=100")]))
