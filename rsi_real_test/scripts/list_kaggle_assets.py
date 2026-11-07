"""Public dataset file inventory with pagination and explicit geometry subset."""
from collections import Counter
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]/"data/metadata"
BASE="https://www.kaggle.com/api/v1/datasets/list/anonymous336/physical-field-material-parameter-5000"


def run():
    files=[];token="";page=0
    while True:
        query={"pageSize":10000}
        if token:query["pageToken"]=token
        url=BASE+"?"+urllib.parse.urlencode(query)
        with urllib.request.urlopen(url,timeout=60) as r:
            d=json.load(r)
        if d.get("hasErrorMessage"):
            raise RuntimeError(d.get("errorMessage"))
        batch=d.get("datasetFiles",[])
        files.extend({"name":x["name"],"bytes":x["totalBytes"]} for x in batch)
        page+=1
        if page % 25 == 0:
            print(json.dumps({"page":page,"files":len(files),"batch":len(batch)}),flush=True)
        next_token=d.get("nextPageToken")
        if not next_token:break
        if next_token==token or not batch:raise RuntimeError("Pagination stopped advancing")
        token=next_token
    if len({x["name"] for x in files})!=len(files):raise ValueError("Duplicate names")
    geometry=[x for x in files if x["name"].lower().endswith((".ply",".obj",".glb",".stl"))]
    result={"url":"https://www.kaggle.com/datasets/anonymous336/physical-field-material-parameter-5000",
        "files":len(files),"total_bytes":sum(x["bytes"] for x in files),
        "geometry_count":len(geometry),"geometry_bytes":sum(x["bytes"] for x in geometry),
        "geometry":geometry,"top_level_counts":dict(Counter(x["name"].split('/')[0] for x in files))}
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/"kaggle_file_inventory.json").write_text(json.dumps(files,indent=2))
    (ROOT/"kaggle_geometry_inventory.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="geometry"},indent=2))


if __name__=="__main__":run()
