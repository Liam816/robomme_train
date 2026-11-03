"""Resume the byte-identical pi05 mirror; verify official GCS MD5/CRC32C."""
from pathlib import Path
import base64,concurrent.futures as cf,hashlib,json,os,re,subprocess,threading,time,urllib.parse
R=Path("/data/heping/robomme_musa")
items=json.loads((R/"gcs_manifest.json").read_text())["items"]
dest=R/"assets/pi05_base/params";spool=R/"assets/download_spool"
spool.mkdir(parents=True,exist_ok=True)
lock=threading.Lock();CHUNK=8*1024**2
def checksum(path,item):
    if "md5Hash" not in item:
        value=subprocess.check_output([str(R/"crc32c_file"),str(path)],text=True).strip()
        return base64.b64encode(bytes.fromhex(value)).decode()
    h=hashlib.md5()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
    return base64.b64encode(h.digest()).decode()
def expected(item):return item.get("md5Hash",item["crc32c"])
def atomic(path,value):
    temp=path.with_name(path.name+".tmp");temp.write_text(json.dumps(value));temp.replace(path)
def header_ok(text,start,end,size):
    return bool(re.search(r"(?im)^content-range:\s*bytes "+str(start)+"-"+str(end)+"/"+str(size)+r"\s*$",text))
states={};groups=[]
for item in items:
    rel=item["name"].removeprefix("checkpoints/pi05_base/params/")
    target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);size=int(item["size"])
    if target.exists() and target.stat().st_size==size and checksum(target,item)==expected(item):continue
    stage=target.with_name(target.name+".ranges");journal=target.with_name(target.name+".ranges.json")
    if journal.exists():
        state=json.loads(journal.read_text());assert state["md5"]==expected(item) and state["size"]==size
        if state.get("chunk",32*1024**2)!=CHUNK:
            oldsize=state.get("chunk",32*1024**2);done={}
            for key,digest in state["done"].items():
                start=int(key)
                with stage.open("rb") as f:f.seek(start);data=f.read(min(oldsize,size-start))
                if hashlib.sha256(data).hexdigest()!=digest:continue
                for offset in range(0,len(data),CHUNK):
                    part=data[offset:offset+CHUNK]
                    if len(part)==min(CHUNK,size-start-offset):done[str(start+offset)]=hashlib.sha256(part).hexdigest()
            state["done"]=done
    else:
        state={"md5":expected(item),"size":size,"done":{}}
        with stage.open("wb") as f:f.truncate(size)
    state["chunk"]=CHUNK
    # Recover old 32 MiB HTTP response prefixes, before reusing their spool names.
    for start in range(0,size,32*1024**2):
        key=hashlib.sha256((str(target)+str(start)).encode()).hexdigest()[:20]
        part=spool/(key+".part");head=spool/(key+".headers")
        if not part.exists() or not head.exists():continue
        text=head.read_text();end=min(size,start+32*1024**2)-1
        if header_ok(text,start,end,size) and 0<part.stat().st_size<=end-start+1:
            data=part.read_bytes()
            fd=os.open(stage,os.O_WRONLY)
            try:os.pwrite(fd,data,start);os.fsync(fd)
            finally:os.close(fd)
            for offset in range(0,len(data),CHUNK):
                block=data[offset:offset+CHUNK]
                if len(block)==min(CHUNK,size-start-offset):state["done"][str(start+offset)]=hashlib.sha256(block).hexdigest()
        archive=R/"assets/previous_spool";archive.mkdir(exist_ok=True)
        part.replace(archive/part.name);head.replace(archive/head.name)
    atomic(journal,state)
    states[str(target)]=(item,stage,journal,state)
    group=[]
    for start in range(0,size,CHUNK):
        end=min(size,start+CHUNK)-1
        if str(start) in state["done"]:
            with stage.open("rb") as f:f.seek(start);data=f.read(end-start+1)
            if hashlib.sha256(data).hexdigest()==state["done"][str(start)]:continue
            del state["done"][str(start)]
        group.append((str(target),start,end))
    groups.append(group)
def download(job):
    target,start,end=job;item,stage,journal,state=states[target]
    key=hashlib.sha256((target+str(start)).encode()).hexdigest()[:20]
    part=spool/(key+".part");header=spool/(key+".headers")
    incoming=spool/(key+".incoming");ih=spool/(key+".incoming.headers")
    if part.exists() and not(header.exists() and header_ok(header.read_text(),start,end,state["size"])):
        part.rename(part.with_suffix(".untrusted"));header.unlink(missing_ok=True)
    url="https://modelscope.cn/models/hairuoliu/pi05_base/resolve/171d7c8641a306698b2f82eea84325997de0f34f/"+item["name"].removeprefix("checkpoints/pi05_base/")
    for attempt in range(100):
        have=part.stat().st_size if part.exists() else 0
        if have==end-start+1:break
        if have>end-start+1:raise ValueError("Oversize spool")
        begin=start+have
        ip=["104.154.124.27","142.250.73.91","142.250.9.207"][attempt%3]
        cmd=["curl","-4","--fail","--silent","--show-error","--location","--connect-timeout","12","--max-time","180",
             "--range",f"{begin}-{end}",
             "--dump-header",str(ih),"--output",str(incoming),url]
        c=subprocess.run(cmd,capture_output=True,text=True)
        text=ih.read_text() if ih.exists() else ""
        if incoming.exists() and 0<incoming.stat().st_size<=end-begin+1 and header_ok(text,begin,end,state["size"]):
            with part.open("ab") as out,incoming.open("rb") as source:
                for b in iter(lambda:source.read(1024**2),b""):out.write(b)
                out.flush();os.fsync(out.fileno())
            header.write_text(f"Content-Range: bytes {start}-{end}/{state['size']}\n")
        incoming.unlink(missing_ok=True);ih.unlink(missing_ok=True)
        if c.returncode and (not part.exists() or part.stat().st_size==have):time.sleep(min(attempt+1,10))
    else:raise RuntimeError("Unfinished range "+target+f" {start}-{end}")
    data=part.read_bytes();assert len(data)==end-start+1
    fd=os.open(stage,os.O_WRONLY)
    try:
        offset=0
        while offset<len(data):offset+=os.pwrite(fd,data[offset:],start+offset)
        os.fsync(fd)
    finally:os.close(fd)
    with lock:
        state["done"][str(start)]=hashlib.sha256(data).hexdigest();atomic(journal,state)
    part.unlink();header.unlink()
    return len(data)
jobs=[g[i] for i in range(max(map(len,groups),default=0)) for g in groups if i<len(g)]
print("RESUMED_JOBS",len(jobs),"RECOVERED_CHUNKS",sum(len(s[3]["done"]) for s in states.values()),flush=True)
done=0
with cf.ThreadPoolExecutor(max_workers=32) as pool:
    futures=[pool.submit(download,j) for j in jobs]
    for f in cf.as_completed(futures):
        done+=f.result();print("VERIFIED_RANGE_BYTES",done,flush=True)
for target,(item,stage,journal,state) in states.items():
    assert checksum(stage,item)==expected(item),target
    stage.replace(target);journal.unlink();print("OBJECT_CHECKSUM_OK",target,flush=True)
tokenizer=R/"assets/paligemma_tokenizer.model"
c=subprocess.run(["curl","--fail","--silent","--show-error","--retry","6","--max-time","600","--output",str(tokenizer),"https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"],capture_output=True,text=True)
if c.returncode:raise RuntimeError(c.stderr)
directory=R/"assets/big_vision";directory.mkdir(exist_ok=True)
link=directory/"paligemma_tokenizer.model"
if not link.exists():link.symlink_to(tokenizer)
atomic(R/"assets/download_complete.json",{"bytes":sum(int(i["size"]) for i in items),"objects":len(items),
 "verification":"all official GCS MD5 or CRC32C values","mirror":"hairuoliu/pi05_base@171d7c8641a306698b2f82eea84325997de0f34f",
 "tokenizer_sha256":hashlib.sha256(tokenizer.read_bytes()).hexdigest()})
print("OFFICIAL_ASSETS_COMPLETE",flush=True)
