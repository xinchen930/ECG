# PPG Data Quality Analysis Report

**Generated**: 2026-02-05
**Source**: `eval_results/ppg_analysis_all_samples.csv`

## Summary

| Quality | Count | Percentage | Avg HR Error (BPM) | Max HR Error |
|---------|-------|------------|-------------------|--------------|
| **good** | 80 | 81.6% | 3.5 | 10 |
| moderate | 8 | 8.2% | 13.0 | 16 |
| **poor** | 10 | 10.2% | 49.2 | 80 |
| **Total** | **98** | 100% | - | - |

## Quality Classification Criteria

- **good**: HR error ≤ 10 BPM
- **moderate**: 10 < HR error ≤ 20 BPM
- **poor**: HR error > 20 BPM (unusable for HR-based tasks)

## Poor Samples (Recommended for Exclusion)

These 10 samples have severe heart rate detection failures and should likely be excluded from training:

| Pair | User | PPG HR | GT HR | HR Error | SNR | Notes |
|------|------|--------|-------|----------|-----|-------|
| pair_0003 | wjy | 54.0 | 96 | 42.0 | 0.196 | Default HR value |
| pair_0004 | czq | 54.0 | 90 | 36.0 | 0.290 | Default HR value |
| pair_0007 | czq | 54.0 | 90 | 36.0 | 0.252 | Default HR value |
| pair_0011 | fzq | 54.0 | 134 | 80.0 | 0.312 | Default HR value |
| pair_0026 | wjy | 162.0 | 83 | 79.0 | 0.161 | Doubled HR (harmonic) |
| pair_0037 | fcy | 54.0 | 94 | 40.0 | 0.201 | Default HR value |
| pair_0038 | czq | 54.0 | 110 | 56.0 | 0.477 | Default HR value |
| pair_0053 | fcy | 54.0 | 105 | 51.0 | 0.171 | Default HR value |
| pair_0079 | czq | 54.0 | 96 | 42.0 | 0.409 | Default HR value |
| pair_0086 | fcy | 54.0 | 84 | 30.0 | 0.431 | Default HR value |

**Key observation**: 9/10 poor samples have PPG HR = 54.0 BPM, which appears to be a fallback value when the algorithm fails to detect valid peaks. All poor samples have low SNR (< 0.5).

## Moderate Samples

| Pair | User | PPG HR | GT HR | HR Error | SNR |
|------|------|--------|-------|----------|-----|
| pair_0005 | fzq | 162.0 | 151 | 11.0 | 0.623 |
| pair_0023 | syw | 72.0 | 88 | 16.0 | 0.413 |
| pair_0024 | syw | 84.0 | 98 | 14.0 | 0.230 |
| pair_0034 | fzq | 78.0 | 92 | 14.0 | 0.697 |
| pair_0039 | czq | 78.0 | 90 | 12.0 | 0.529 |
| pair_0041 | czq | 114.0 | 100 | 14.0 | 0.201 |
| pair_0069 | czq | 84.0 | 96 | 12.0 | 0.326 |
| pair_0070 | lrk | 96.0 | 85 | 11.0 | 0.557 |

## Quality by User

| User | Total Samples | Poor | Moderate | Good | Poor Rate |
|------|---------------|------|----------|------|-----------|
| czq | 13 | 4 | 3 | 6 | 31% |
| fcy | 15 | 3 | 0 | 12 | 20% |
| wjy | 13 | 2 | 0 | 11 | 15% |
| fzq | 24 | 1 | 2 | 21 | 4% |
| syw | 12 | 0 | 2 | 10 | 0% |
| wcp | 7 | 0 | 0 | 7 | 0% |
| nxs | 6 | 0 | 0 | 6 | 0% |
| lrk | 5 | 0 | 1 | 4 | 0% |
| fhy | 3 | 0 | 0 | 3 | 0% |

**Observation**: User `czq` has significantly higher poor sample rate (31%), possibly due to data collection issues (finger pressure, lighting, skin tone).

## Good Samples HR Error Distribution

| HR Error Range | Count | Percentage |
|----------------|-------|------------|
| 0-3 BPM | 37 | 46.3% |
| 3-6 BPM | 20 | 25.0% |
| 6-10 BPM | 22 | 27.5% |
| ≥10 BPM | 1 | 1.3% |

## Recommendations

### For End-to-End Models (Scheme E/F/G)

1. **Conservative**: Use only `good` samples (80 samples → ~850 windows)
2. **Moderate**: Use `good` + `moderate` (88 samples → ~930 windows)
3. **Experimental**: Try all samples but monitor per-sample loss

### For PPG-based Features

- **Must exclude** poor samples - the PPG signal is likely corrupted
- Consider excluding moderate samples for stricter quality control

### Implementation

**方式一：命令行参数（推荐，灵活切换）**

```bash
# 只用高质量样本 (80个)
python models/train.py --config configs/scheme_e.yaml --quality-filter good

# 排除 poor 样本 (88个)
python models/train.py --config configs/scheme_e.yaml --quality-filter good,moderate

# 用全部数据 (98个)
python models/train.py --config configs/scheme_e.yaml --quality-filter all
```

**方式二：在 config 文件中设置**

```yaml
data:
  quality_filter: "good"          # Only use good samples
  # quality_filter: "good,moderate"  # Include moderate
  # quality_filter: null            # Use all samples
  quality_csv: eval_results/ppg_analysis_all_samples.csv
```

> 命令行参数会覆盖 config 文件中的设置

## Poor Sample Pairs to Exclude

```
pair_0003, pair_0004, pair_0007, pair_0011, pair_0026,
pair_0037, pair_0038, pair_0053, pair_0079, pair_0086
```
