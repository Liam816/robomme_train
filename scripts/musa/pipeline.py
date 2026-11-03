"""Fail-fast, resumable asset/validation/cache/short-training coordinator."""
from pathlib import Path
import base64,fcntl,hashlib,json,os,subprocess,time

ROOT=Path('/data/heping/robomme_musa')
LAUNCH=ROOT/'repo/scripts/run_musa_percmem.sh'
lock=(ROOT/'pipeline.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'pid':os.getpid(),'physical_gpus':[4,5,6,7],'started_at':time.time()}
def update(**values):
    state.update(values)
    p=ROOT/'pipeline.status.json.tmp'
    p.write_text(json.dumps(state,indent=2));p.replace(ROOT/'pipeline.status.json')
def run(stage,args):
    update(stage=stage)
    with (ROOT/(stage+'.log')).open('a') as log:
        log.write('\nPIPELINE_START '+str(time.time())+'\n');log.flush()
        command=['bash',str(LAUNCH),*args]
        if stage=='export':
            available=sorted(os.sched_getaffinity(0)&set(range(64,72)))
            command=['taskset','-c',','.join(map(str,available)),*command]
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        update(child_pid=child.pid)
        code=child.wait()
        if code:raise RuntimeError(stage+' exited '+str(code))
def verify_tokenizer():
    update(stage='verifying_official_tokenizer')
    path=ROOT/'assets/tokenizer_gcs_metadata.json'
    if not path.exists():
        temp=path.with_suffix('.tmp')
        subprocess.run(['curl','-4','-fLsS','--retry','3','--max-time','60','-o',str(temp),
            'https://storage.googleapis.com/storage/v1/b/big_vision/o/paligemma_tokenizer.model'],check=True)
        temp.replace(path)
    meta=json.loads(path.read_text())
    assert meta['bucket']=='big_vision' and meta['name']=='paligemma_tokenizer.model'
    data=(ROOT/'assets/paligemma_tokenizer.model').read_bytes()
    assert len(data)==int(meta['size'])
    assert base64.b64encode(hashlib.md5(data).digest()).decode()==meta['md5Hash']
    (ROOT/'assets/tokenizer_verification.json').write_text(json.dumps({
        'passed':True,'generation':meta['generation'],'size':len(data),
        'official_md5_matched':True,'sha256':hashlib.sha256(data).hexdigest()},indent=2))
try:
    update(stage='waiting_for_verified_weights')
    while not (ROOT/'assets/download_complete.json').exists():time.sleep(10)
    verify_tokenizer()
    if not (ROOT/'assets/export_complete.json').exists():run('export',['export'])
    report=ROOT/'production_validation.json'
    if not report.exists() or not json.loads(report.read_text()).get('passed'):
        run('validate_production',['validate-production'])
    if not (ROOT/'data/percmem_cache/meta/native_complete.json').exists():
        run('extract',['extract'])
    run('short_train',['train','--steps','3','--run-name','musa_smoke_verified'])
    update(stage='complete',finished_at=time.time())
except BaseException as error:
    update(stage='failed',error=repr(error),finished_at=time.time())
    raise
