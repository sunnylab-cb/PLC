from pathlib import Path

def rep(s, old, new, path):
    if old not in s:
        raise SystemExit(f'MISSING in {path}: {old[:160]!r}')
    return s.replace(old, new, 1)

def patch_train(path_s: str):
    path = Path(path_s)
    s = path.read_text()
    s = rep(s,
"""    value_clip: float = 0.0,
    value_clip_param_tokens: Tuple[str, ...] = (),
) -> float:
""",
"""    value_clip: float = 0.0,
    value_clip_param_tokens: Tuple[str, ...] = (),
    value_clip_groups: Optional[List[Dict]] = None,
) -> float:
""", path)
    s = rep(s,
"""        if (
            value_clip > 0
            and value_clip_param_tokens
            and any(tok in name for tok in value_clip_param_tokens)
        ):
            grad_values.clamp_(min=-float(value_clip), max=float(value_clip))
""",
"""        clip_values = []
        if (
            value_clip > 0
            and value_clip_param_tokens
            and any(tok in name for tok in value_clip_param_tokens)
        ):
            clip_values.append(float(value_clip))
        for group in value_clip_groups or ():
            group_value = float(group.get('value', 0.0) or 0.0)
            group_tokens = tuple(group.get('tokens', ()) or ())
            if group_value > 0 and group_tokens and any(tok in name for tok in group_tokens):
                clip_values.append(group_value)
        if clip_values:
            clip_value = min(clip_values)
            grad_values.clamp_(min=-clip_value, max=clip_value)
""", path)
    s = rep(s,
"""    zero_open_lr_mult: float = 1.0,
    zero_open_eps: Optional[float] = None,
) -> Tuple[List[Dict], Dict[str, object]]:
""",
"""    zero_open_lr_mult: float = 1.0,
    zero_open_eps: Optional[float] = None,
    ligand_lr_mult: float = 1.0,
    ligand_lr_tokens: Optional[Tuple[str, ...]] = None,
) -> Tuple[List[Dict], Dict[str, object]]:
""", path)
    s = rep(s,
"""    decay_params = []
    no_decay_params = []
    zero_open_params = []
    decay_names = []
    no_decay_names = []
    zero_open_names = []
    for name, p in model.named_parameters():
""",
"""    decay_params = []
    no_decay_params = []
    zero_open_params = []
    ligand_decay_params = []
    ligand_no_decay_params = []
    decay_names = []
    no_decay_names = []
    zero_open_names = []
    ligand_decay_names = []
    ligand_no_decay_names = []
    ligand_lr_tokens = tuple(ligand_lr_tokens or ())
    for name, p in model.named_parameters():
""", path)
    s = rep(s,
"""        is_zero_open = any(tok in name for tok in ZERO_OPEN_PARAM_TOKENS)
        is_no_decay = (
""",
"""        is_zero_open = any(tok in name for tok in ZERO_OPEN_PARAM_TOKENS)
        is_ligand_boost = (
            ligand_lr_mult != 1.0
            and ligand_lr_tokens
            and any(tok in name for tok in ligand_lr_tokens)
        )
        is_no_decay = (
""", path)
    s = rep(s,
"""        if is_zero_open:
            zero_open_params.append(p)
            zero_open_names.append(name)
        elif is_no_decay:
            no_decay_params.append(p)
            no_decay_names.append(name)
        else:
            decay_params.append(p)
            decay_names.append(name)
""",
"""        if is_zero_open:
            zero_open_params.append(p)
            zero_open_names.append(name)
        elif is_ligand_boost and is_no_decay:
            ligand_no_decay_params.append(p)
            ligand_no_decay_names.append(name)
        elif is_ligand_boost:
            ligand_decay_params.append(p)
            ligand_decay_names.append(name)
        elif is_no_decay:
            no_decay_params.append(p)
            no_decay_names.append(name)
        else:
            decay_params.append(p)
            decay_names.append(name)
""", path)
    s = rep(s,
"""    groups: List[Dict] = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    if zero_open_params:
""",
"""    groups: List[Dict] = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    if ligand_decay_params:
        groups.append({
            'params': ligand_decay_params,
            'weight_decay': weight_decay,
            'lr': float(base_lr) * float(ligand_lr_mult) if base_lr is not None else None,
        })
    if ligand_no_decay_params:
        groups.append({
            'params': ligand_no_decay_params,
            'weight_decay': 0.0,
            'lr': float(base_lr) * float(ligand_lr_mult) if base_lr is not None else None,
        })
    if zero_open_params:
""", path)
    s = rep(s,
"""        'zero_open_count': len(zero_open_params),
        'decay_names': decay_names,
        'no_decay_names': no_decay_names,
        'zero_open_names': zero_open_names,
""",
"""        'zero_open_count': len(zero_open_params),
        'ligand_decay_count': len(ligand_decay_params),
        'ligand_no_decay_count': len(ligand_no_decay_params),
        'decay_names': decay_names,
        'no_decay_names': no_decay_names,
        'zero_open_names': zero_open_names,
        'ligand_decay_names': ligand_decay_names,
        'ligand_no_decay_names': ligand_no_decay_names,
""", path)
    s = rep(s,
"""    grad_value_clip: float = 0.0,        # V12: finite spike guard for selected params
    grad_value_clip_param_tokens: Optional[List[str]] = None,
) -> Dict[str, float]:
""",
"""    grad_value_clip: float = 0.0,        # V12: finite spike guard for selected params
    grad_value_clip_param_tokens: Optional[List[str]] = None,
    grad_value_clip_groups: Optional[List[Dict]] = None,
) -> Dict[str, float]:
""", path)
    s = rep(s,
"""                        value_clip=grad_value_clip,
                        value_clip_param_tokens=tuple(grad_value_clip_param_tokens or ()),
                    )
""",
"""                        value_clip=grad_value_clip,
                        value_clip_param_tokens=tuple(grad_value_clip_param_tokens or ()),
                        value_clip_groups=grad_value_clip_groups,
                    )
""", path)
    s = rep(s,
"""    zero_open_eps = float(train_cfg.get('zero_open_adamw_eps', 1e-3))
    param_groups, param_group_info = build_adamw_param_groups(
        model,
        train_cfg['weight_decay'],
        base_lr=float(train_cfg['learning_rate']),
        zero_open_lr_mult=zero_open_lr_mult,
        zero_open_eps=zero_open_eps,
    )
""",
"""    zero_open_eps = float(train_cfg.get('zero_open_adamw_eps', 1e-3))
    ligand_lr_mult = float(train_cfg.get('ligand_encoder_lr_mult', 1.0))
    ligand_lr_tokens = tuple(train_cfg.get('ligand_lr_param_tokens', [
        'ligand_encoder', 'unimol_ligand_proj', 'unimol_ligand_alpha',
    ]))
    param_groups, param_group_info = build_adamw_param_groups(
        model,
        train_cfg['weight_decay'],
        base_lr=float(train_cfg['learning_rate']),
        zero_open_lr_mult=zero_open_lr_mult,
        zero_open_eps=zero_open_eps,
        ligand_lr_mult=ligand_lr_mult,
        ligand_lr_tokens=ligand_lr_tokens,
    )
""", path)
    s = rep(s,
"""            f"no_decay={param_group_info['no_decay_count']}, "
            f"zero_open={param_group_info['zero_open_count']} "
            f"(lr_mult={zero_open_lr_mult}, eps={zero_open_eps})"
""",
"""            f"no_decay={param_group_info['no_decay_count']}, "
            f"ligand_decay={param_group_info['ligand_decay_count']}, "
            f"ligand_no_decay={param_group_info['ligand_no_decay_count']} "
            f"(ligand_lr_mult={ligand_lr_mult}), "
            f"zero_open={param_group_info['zero_open_count']} "
            f"(lr_mult={zero_open_lr_mult}, eps={zero_open_eps})"
""", path)
    s = rep(s,
"""            grad_value_clip_param_tokens=train_cfg.get(
                'grad_value_clip_param_tokens',
                ['esm_proj.0.weight'],
            ),
            unimol_cache=unimol_cache,
""",
"""            grad_value_clip_param_tokens=train_cfg.get(
                'grad_value_clip_param_tokens',
                ['esm_proj.0.weight'],
            ),
            grad_value_clip_groups=train_cfg.get('grad_value_clip_groups', []),
            unimol_cache=unimol_cache,
""", path)
    path.write_text(s)
    print(f'patched train {path}')

for f in ['scripts/train/train_v12.py', 'scripts/train/train_v12m1.py']:
    patch_train(f)
