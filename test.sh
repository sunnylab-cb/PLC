# 1) 生成3份对照配置
  python - <<'PY'
  import yaml, copy
  from pathlib import Path
  base = yaml.safe_load(open('configs/pocket_gvp_v9.yaml'))
  out = Path('configs/io_sweep'); out.mkdir(parents=True, exist_ok=True)

      c['experiment']['name'] = tag
      c['training']['num_workers'] = nw
      c['training']['prefetch_factor'] = pf
      c['training']['pin_memory'] = True
      c['training']['enable_profiling'] = True   # 先开着做对照
      c['training']['num_epochs'] = 1            # 快速比较
      c['data']['max_seq_len'] = msl
      with open(out / f'{tag}.yaml', 'w') as f:
          yaml.safe_dump(c, f, sort_keys=False)

  dump('ioA_w16_pf2_l1500', 16, 2, 1500)
  dump('ioB_w24_pf1_l1500', 24, 1, 1500)
  dump('ioC_w16_pf2_l1024', 16, 2, 1024)
  print('written:', [p.name for p in out.glob('*.yaml')])
  PY

  # 2) 依次运行（每组最多25分钟，够看稳定性趋势）
  for c in configs/io_sweep/ioA_w16_pf2_l1500.yaml \
           configs/io_sweep/ioB_w24_pf1_l1500.yaml \
           configs/io_sweep/ioC_w16_pf2_l1024.yaml; do
    echo "===== RUN $c ====="
    timeout 25m /home/husrcf/anaconda3/envs/AIAA/bin/python scripts/train/train_v9m2.py --config "$c"
  done
