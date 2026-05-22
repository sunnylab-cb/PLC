from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
runs = {
    'v12m2_strong_gate': Path('outputs/hpc_sync/pocket_gvp_fusion_calib_v12m2_20260521_220112'),
    'v12m3_pair_old128': Path('outputs/hpc_sync/pocket_gvp_fusion_pair_evidence_v12m3_20260521_220558'),
}
keys = ['Val/f1_c0','Val/dice_c0','Val/iou_c0','Val/epoch_loss','Val/precision_c0','Val/recall_c0','Train/epoch_loss','Gradients/grad_norm_ligand_encoder','Gradients/grad_norm_interaction_blocks','Gradients/grad_norm_interaction_tokens','Gradients/grad_norm_esm_proj','Gradients/grad_norm_gvp_proj','Gradients/grad_norm_pair_evidence','Gradients/grad_norm_pair_int_pool_proj','Health/pair_score_max_mean','Health/pair_entropy_mean','Health/pair_context_norm']
allv={}
for name,root in runs.items():
    ev=sorted((root/'runs').glob('events.out.tfevents.*'))[-1]
    ea=EventAccumulator(str(ev), size_guidance={'scalars':0}); ea.Reload()
    allv[name]={}
    tags=set(ea.Tags().get('scalars',[]))
    for k in keys:
        if k in tags:
            vals=ea.Scalars(k)
            allv[name][k]={v.step:v.value for v in vals}
            allv[name][k+'_last']=(vals[-1].step, vals[-1].value)
            best=max(vals,key=lambda x:x.value) if any(s in k for s in ['f1_c0','dice_c0','iou_c0','precision','recall']) else min(vals,key=lambda x:x.value)
            allv[name][k+'_best']=(best.step,best.value)

print('Same-epoch comparison at epoch 8')
for k in ['Val/f1_c0','Val/dice_c0','Val/iou_c0','Val/epoch_loss','Val/precision_c0','Val/recall_c0','Train/epoch_loss']:
    a=allv['v12m2_strong_gate'].get(k,{}).get(8)
    b=allv['v12m3_pair_old128'].get(k,{}).get(8)
    if a is not None and b is not None:
        delta=b-a
        pct=delta/(abs(a)+1e-12)*100
        print(f'{k:35s} v12m2={a:.6f} v12m3={b:.6f} delta={delta:+.6f} ({pct:+.2f}%)')
print('\nLatest / best')
for name in runs:
    print('\n',name)
    for k in keys:
        if k+'_last' in allv[name]:
            lstep,lval=allv[name][k+'_last']; bstep,bval=allv[name][k+'_best']
            print(f'{k:35s} last@{lstep}={lval:.6f} best@{bstep}={bval:.6f}')
print('\nGradient ratios at latest scalar step')
for name in runs:
    d=allv[name]
    print('\n',name)
    def last(k): return d.get(k+'_last',(None,0.0))[1]
    ligand=last('Gradients/grad_norm_ligand_encoder')
    inter=last('Gradients/grad_norm_interaction_blocks')
    tok=last('Gradients/grad_norm_interaction_tokens')
    esm=last('Gradients/grad_norm_esm_proj')
    gvp=last('Gradients/grad_norm_gvp_proj')
    pair=last('Gradients/grad_norm_pair_evidence')
    print(f'ligand/esm={ligand/(esm+1e-12):.3f}, inter/esm={inter/(esm+1e-12):.3f}, tokens/esm={tok/(esm+1e-12):.3f}, gvp/esm={gvp/(esm+1e-12):.3f}, pair/esm={pair/(esm+1e-12):.3f}')
