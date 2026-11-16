import hashlib, json, os, subprocess, sys, time
from pathlib import Path
root=Path(r'E:\digital_twin_armigo')
sp=Path(r'C:\Users\Jiang\AppData\Local\Temp\claude\cfc6378ac91412cf\94b28ae0-0a02-42e3-9bb5-e60d7ce00e76\scratchpad')
out=root/'results/verification-2026-11-17'
data=out/'data'
data.mkdir(exist_ok=True)
base=sp/'baseline'
base_script=base/'scripts/sim/so101_sort_sim.py'
head=subprocess.check_output(['git','show','HEAD:scripts/sim/so101_sort_sim.py'],cwd=root)
assert hashlib.sha256(head.replace(b'\r\n',b'\n')).digest()==hashlib.sha256(base_script.read_bytes().replace(b'\r\n',b'\n')).digest(), 'baseline differs from HEAD'
env=dict(os.environ, MUJOCO_GL='glfw')
times=[]
def run(label, cwd, script, args):
    cmd=[sys.executable,str(script),*map(str,args)]
    start=time.perf_counter()
    with (out/(label+'.log')).open('w',encoding='utf-8') as f:
        p=subprocess.run(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
    rec={'label':label,'seconds':time.perf_counter()-start,'returncode':p.returncode,'command':cmd}
    times.append(rec)
    (out/'timings.json').write_text(json.dumps(times,indent=2),encoding='utf-8')
    print(label,round(rec['seconds'],3),'exit',p.returncode,flush=True)
    if p.returncode: raise RuntimeError((label,(out/(label+'.log')).read_text()[-3000:]))
cam=root/'config/sim/top_camera_joint.json'
prior=root/'config/sim/prior_grasp.json'
for label,cwd,workers in [('sim_baseline',base,1),('sim_current',root,1),('sim_workers2',root,2)]:
    run(label,cwd,cwd/'scripts/sim/so101_sort_sim.py',['--menagerie',cwd/'assets/so101_mjcf','simulate','--camera',cam,'--prior',prior,'--out',data/label,'--episodes',3,'--seed',0,'--workers',workers])
for label,cwd,workers in [('render_baseline',base,1),('render_baseline_repeat',base,1),('render_current',root,1),('render_workers2',root,2)]:
    run(label,cwd,cwd/'scripts/sim/so101_sort_sim.py',['--menagerie',cwd/'assets/so101_mjcf','render','--camera',cam,'--sim',data/'sim_baseline','--out',data/label,'--views',1,'--seed',1,'--stride',10,'--workers',workers])
for label,a,b,tol in [('simulation','sim_baseline','sim_current',0),('simulation_workers','sim_current','sim_workers2',0),('render_repeat','render_baseline','render_baseline_repeat',1),('render','render_baseline','render_current',1),('render_workers','render_current','render_workers2',1)]:
    run('compare_'+label,root,root/'scripts/sim/compare_outputs.py',[data/a,data/b,'--rgb-atol',tol,'--report',out/('compare_'+label+'.json')])
