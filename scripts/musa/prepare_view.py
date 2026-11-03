"""Create a separate canonical LeRobot view. Never edit the original data."""
from pathlib import Path
import json, shutil
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path("/data/heping/robomme_musa")
SOURCE=Path("/home/mccuseradmin/heping/data/datasets/yinchenghust/robomme_lerobot")
DEST=ROOT/"data/lerobot_view"
if (DEST/"meta/view_complete.json").exists():
    print("VIEW_ALREADY_COMPLETE")
    raise SystemExit(0)
DEST.mkdir(parents=True,exist_ok=True)
mapping={"observation.state":"state","action":"actions","frame_index":"step_idx","is_video_demo":"is_demo"}
cameras={"observation.images.front":"image","observation.images.wrist":"wrist_image"}
info=json.loads((SOURCE/"meta/info.json").read_text())
tables=[pq.read_table(p) for p in sorted((SOURCE/"data").glob("*/*.parquet"))]
table=pa.concat_tables(tables)
episodes=[]
for p in sorted((SOURCE/"meta/episodes").glob("*/*.parquet")):
    episodes.extend(pq.read_table(p).to_pylist())
episodes.sort(key=lambda e:e["episode_index"])
is_demo=table["is_video_demo"].to_numpy(zero_copy_only=False)
starts=np.empty(len(table),np.int32)
for e in episodes:
    begin,end=e["dataset_from_index"],e["dataset_to_index"]
    demo=is_demo[begin:end]; count=int(demo.sum())
    if not np.array_equal(demo,np.arange(end-begin)<count):
        raise ValueError("Demo mask must be a contiguous episode prefix")
    starts[begin:end]=count
names=[mapping.get(n,n) for n in table.column_names]
table=table.rename_columns(names).append_column("exec_start_idx",pa.array(starts))
p=DEST/"data/chunk-000/file-000.parquet"
p.parent.mkdir(parents=True,exist_ok=True)
pq.write_table(table,p)
for e in episodes:
    for old,new in cameras.items():
        for key in list(e):
            if key.startswith("videos/"+old+"/"):
                e[key.replace("videos/"+old+"/","videos/"+new+"/",1)]=e.pop(key)
p=DEST/"meta/episodes/chunk-000/file-000.parquet"
p.parent.mkdir(parents=True,exist_ok=True)
pq.write_table(pa.Table.from_pylist(episodes),p)
(DEST/"videos").mkdir(exist_ok=True)
for old,new in cameras.items():
    target=DEST/"videos"/new
    if not target.exists(): target.symlink_to(SOURCE/"videos"/old,target_is_directory=True)
info["features"]={cameras.get(mapping.get(k,k),mapping.get(k,k)):v for k,v in info["features"].items()}
info["features"]["exec_start_idx"]={"dtype":"int32","shape":[1],"names":None}
(DEST/"meta/info.json").write_text(json.dumps(info,indent=2))
for name in ["tasks.parquet","subtasks.parquet"]:
    shutil.copyfile(SOURCE/"meta"/name,DEST/"meta"/name)
report={"source":str(SOURCE),"view":str(DEST),"rows":len(table),"episodes":len(episodes),
        "execution_samples":int((~is_demo).sum()),"fps":info["fps"],
        "mapping":mapping,"cameras":cameras,"source_unchanged":True}
(DEST/"meta/view_complete.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
