## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

To overlay two runs on shared axes for an A/B comparison (for example SSD
streaming with and without the expert-bundle sidecar):

```
python3 speed-bench/plot_speed_compare.py \
  speed-bench/no_bundle.csv speed-bench/bundle.csv \
  --label-a "senza bundle" --label-b "con bundle"
```

The first CSV is the baseline (drawn dashed), the second the candidate (drawn
solid). The subtitle reports the candidate's average prefill and generation
delta over the context points the two runs share. Output defaults to
`<candidate>_vs_<baseline>_ts.svg` next to the candidate CSV.
