from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for root in ['outputs/hpc_sync/pocket_gvp_fusion_calib_v12m2_20260521_220112','outputs/hpc_sync/pocket_gvp_fusion_pair_evidence_v12m3_20260521_220558']:
    ev=sorted((Path(root)/'runs').glob('events.out.tfevents.*'))[-1]
    ea=EventAccumulator(str(ev), size_guidance={'scalars':0}); ea.Reload()
    tags=ea.Tags().get('scalars', [])
    print('\nRUN', Path(root).name, 'n_tags', len(tags))
    for tag in tags:
        low=tag.lower()
        if any(s in low for s in ['pair','interaction','grad_norm_interaction','grad_norm_pair','health/interaction']):
            vals=ea.Scalars(tag)
            if vals:
                print(f'{tag}: n={len(vals)} first={vals[0].value:.6g}@{vals[0].step} mid={vals[len(vals)//2].value:.6g}@{vals[len(vals)//2].step} last={vals[-1].value:.6g}@{vals[-1].step}')
