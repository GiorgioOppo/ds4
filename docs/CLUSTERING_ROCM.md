# DeepSeek V4.1: two-machine ROCm cluster

- Two ROCm/gfx1151 machines; tested with 128 GB RAM each.
- Same engine revision and `DeepSeek-V4.1-Flash-Q2.gguf` on both. Each keeps the full GGUF; each machine loads approximately 80.6 GiB of weights into RAM. Engram stays on disk.
- Exactly two machines: one coordinator and one worker. They share attention computation and split the experts equally.
- Cluster mode requires the assigned experts to fit in RAM. SSD expert streaming, DSpark and splitting by `--layers` are not supported.
- Both transports require a reachable TCP control address. Use a trusted network: peer traffic has no authentication or encryption.
- Build both peers with `make strix-halo ROCM_ARCH=gfx1151` after installing any required RoCE headers.
- Run from the engine build directory. Set these variables in **both** terminals; `MODEL` may differ between machines:

```bash
MODEL=/absolute/path/DeepSeek-V4.1-Flash-Q2.gguf
COORD=10.99.0.1       # Coordinator's address on the selected link
CTX=16384
```

## TCP over Ethernet or USB4

- Working Ethernet/IP connection; coordinator TCP port 9911 reachable from the worker. USB4 Ethernet (`thunderbolt_net`) works with the same commands: set `COORD` to the coordinator's USB IP address.
- No RDMA packages required.

```bash
# Coordinator
./ds4-server --rocm -m "$MODEL" --ctx "$CTX" \
  --tensor-parallel --role coordinator --listen "$COORD" 9911 \
  --transport tcp --batched-session 1 --host 127.0.0.1 --port 8080

# Worker, in its own terminal
./ds4 --rocm -m "$MODEL" --ctx "$CTX" \
  --tensor-parallel --role worker --coordinator "$COORD" 9911 \
  --transport tcp
```

## RoCE

- Both hosts need RoCE-capable Ethernet adapters, a working driver and an active Ethernet verbs port. Ordinary Ethernet alone is insufficient.
- Install the runtime/provider packages where inference runs; development headers are needed when building the engine. Package names: [Fedora](https://packages.fedoraproject.org/pkgs/rdma-core/), [Ubuntu](https://packages.ubuntu.com/source/jammy/rdma-core).

```bash
# Fedora, both inference/build environments
sudo dnf install libibverbs libibverbs-utils rdma-core-devel iproute

# Ubuntu/Debian alternative
sudo apt install rdma-core libibverbs1 ibverbs-providers ibverbs-utils libibverbs-dev iproute2
```

```bash
# Both hosts/environments: inspect local device, port and GIDs
ibv_devices
ibv_devinfo -v
rdma link
ulimit -l                      # Locked-memory allowance; at least 16 MiB for RoCE staging
DEV=rocep194s0                 # Replace with this host's verbs device
PORT=1
for file in /sys/class/infiniband/"$DEV"/ports/"$PORT"/gids/*; do
  idx=${file##*/}
  printf '%s  %s  %s  %s\n' "$idx" "$(cat "$file")" \
    "$(cat /sys/class/infiniband/"$DEV"/ports/"$PORT"/gid_attrs/types/"$idx")" \
    "$(cat /sys/class/infiniband/"$DEV"/ports/"$PORT"/gid_attrs/ndevs/"$idx")"
done
GID=1                         # Choose this host's nonzero RoCE v2 GID for the cabled NIC/IP
```

- `rdma link` comes from [Fedora iproute](https://packages.fedoraproject.org/pkgs/iproute/iproute/fedora-rawhide.html) or [Ubuntu iproute2](https://packages.ubuntu.com/jammy/all/iproute2/filelist).
- Device names and GID indexes may differ between hosts. `ibv_devinfo` must show an active port with Ethernet link layer.
- The selected device's `/dev/infiniband/uverbs*` must be accessible. Containers also need its device access, userspace provider and adequate memlock allowance; packages alone do not configure the host NIC.
- Set `COORD` to the coordinator's address on the RoCE Ethernet link.

```bash
# Coordinator
./ds4-server --rocm -m "$MODEL" --ctx "$CTX" \
  --tensor-parallel --role coordinator --listen "$COORD" 9911 \
  --transport rdma --rdma-device "$DEV" --rdma-port "$PORT" --rdma-gid-index "$GID" \
  --batched-session 1 --host 127.0.0.1 --port 8080

# Worker
./ds4 --rocm -m "$MODEL" --ctx "$CTX" \
  --tensor-parallel --role worker --coordinator "$COORD" 9911 \
  --transport rdma --rdma-device "$DEV" --rdma-port "$PORT" --rdma-gid-index "$GID"
```

- RoCE transfers use buffers in system RAM. GPU-direct transfers are not implemented; RCCL is not required.
- Explicit `tcp` or `rdma` fails if unavailable. `auto` negotiates configured RoCE, then TCP at connection setup; no mid-generation fallback.

## Vision and first request

- Add `--vision /absolute/path/DeepSeek-V4.1-Flash-Vision.gguf` to **both** commands. Keep `--ctx` equal on both.
- The HTTP API runs only on the coordinator. Test port 8080 after startup; do not send HTTP to peer port 9911.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Say hello."}],"temperature":0,"max_tokens":64,"thinking":false}'
```


## Measured performance

- Two Framework Desktop systems, 128 GB each, 16-core Strix Halo / `gfx1151`; coordinator Ryzen AI Max+ 395, worker engineering sample `100-000001243-50_Y`.
- Model drives: coordinator SK hynix PC711 1 TB (PCIe 3.0 ×4, ext4); worker Kingston FURY Renegade 2 TB (`SFYRD2000G`, PCIe 4.0 ×4, btrfs). Engram remains disk-backed.
- TCP/RoCE: Intel E810-C QSFP NICs, 100 Gb/s link, MTU 9000. Coordinator NIC negotiated PCIe 3.0 ×4; worker PCIe 4.0 ×4.
- Existing boot settings include `pci=realloc pcie_aspm=off`, in addition to the [GPU-visible memory settings](STRIX_HALO.md#gpu-visible-memory). Their individual performance effect was not isolated.
- Linux `7.2.5-100.fc43.x86_64`, ROCm 10.0 SDK (`10.0.0-4`, HIP `7.15.26333`). TuneD `accelerator-performance`, fans at maximum speed on both machines.
- Same Q2 file, 69,632 allocated context, fresh full prefix, fixed greedy outputs, no DSpark or images. Native `ds4-bench`; startup and a 256-token/128-output warmup excluded. One run per cell; values are **prefill / decode tokens/s**.

| Prompt tokens | Generated tokens | TCP, 100 GbE | RoCE RC, 100 GbE |
|---:|---:|---:|---:|
| 1,024 | 128 | 123.54 / 15.42 | 123.80 / 15.94 |
| 16,384 | 512 | 412.94 / 15.59 | 410.54 / 15.94 |
| 65,536 | 128 | 432.83 / 14.96 | 431.85 / 15.33 |

- A second RoCE 16K/512 run measured **408.25 / 15.92** prefill/decode tokens/s.
- All 129,280 frontier logits and complete printed continuations match across transports at each depth. No OOM; minimum usable RAM in the qualification panel: at least 32.9 GiB. Host zram swap-out was nonzero; these are not zero-swap or cold-cache measurements.
- RoCE device logs and hardware send counters confirm RDMA payloads on both peers. Its TCP control connection is intentional; similar decode rates do not indicate TCP fallback.
- V4.1 CED uses about 8B active parameters/token in prefill and 16B in decode. Full prefixes exercise the decoder-suffix optimization; short appends can follow a different schedule. [Architecture](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/README.md?code=true).
- Results apply to these drives, NIC attachment, profile. Other network adapters have not been tested. Long-running production use has not been tested.

### Appending to an existing prompt

Separately recorded measurements of the unchanged prefill path: same hardware, allocation and warmup; one live session, no generation between frontiers. Values time only the newly appended tokens, in tokens/s.

| Existing → final tokens | Added tokens | TCP, 100 GbE | RoCE |
|---:|---:|---:|---:|
| 4,096 → 8,192 | 4,096 | 219.61 | 218.78 |
| 8,192 → 16,384 | 8,192 | 359.70 | 355.86 |
| 57,344 → 65,536 | 8,192 | 314.92 | 313.65 |

### Reproduce the table

Build the engine on both machines. On the coordinator, build the included TP-only warmup adapter; it links the existing engine objects and leaves `ds4-bench` untouched:

```bash
make strix-halo ROCM_ARCH=gfx1151
bash speed-bench/build-rocm-v41-warmup.sh  # Coordinator only

tuned-adm active                        # Expect accelerator-performance during the workload
tuned-adm verify                        # Verify the applied profile
```

Set this in both terminals after the relevant device setup above:

```bash
MODEL=/absolute/path/DeepSeek-V4.1-Flash-Q2.gguf
COORD=10.99.0.1                         # Coordinator address on the selected link
TRANSPORT=rdma                         # tcp or rdma
DEV=rocep194s0                          # This host's active verbs device
GID=1                                  # This host's matching RoCE v2 GID
LINK=(--tensor-parallel --transport "$TRANSPORT")
case "$TRANSPORT" in
  rdma) LINK+=(--rdma-device "$DEV" --rdma-port 1 --rdma-gid-index "$GID") ;;
esac
```

```bash
# Coordinator: 16K/512; use DEPTH=1024 or 65536 with GEN=128 for the other rows.
DEPTH=16384
GEN=512
./ds4-bench-warm --backend rocm -m "$MODEL" \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start "$DEPTH" --ctx-max "$DEPTH" --ctx-alloc 69632 \
  --gen-tokens "$GEN" --show-output --csv "tp-$TRANSPORT-$DEPTH.csv" \
  --dump-frontier-logits-dir "frontiers-$TRANSPORT-$DEPTH" \
  --role coordinator --listen "$COORD" 19475 "${LINK[@]}"

# Worker: start for each coordinator run.
./ds4 --rocm -m "$MODEL" --ctx 69632 \
  --role worker --coordinator "$COORD" 19475 "${LINK[@]}"
```

- The adapter warms 256 prefix tokens and 128 decode steps, then creates a fresh session before the unchanged native measured loop. A separate short process is not the same warmup procedure.
- Preserve the CSV, full frontier files, printed continuation, revision/build flags, model identity, active power profile and swap/OOM counters. Verify the active power profile before comparing timings; no image-conditioned prefill timings.

For the append table, keep the worker command and replace the coordinator command with:

```bash
# 4K → 8K → 16K, no generated tokens between appends
./ds4-bench-warm --backend rocm -m "$MODEL" \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 4096 --ctx-max 16384 --step-mul 2 --ctx-alloc 69632 \
  --gen-tokens 0 --csv "append-$TRANSPORT.csv" \
  --dump-frontier-logits-dir "append-frontiers-$TRANSPORT" \
  --role coordinator --listen "$COORD" 19475 "${LINK[@]}"
# For 56K → 64K: --ctx-start 57344 --ctx-max 65536 --step-mul 1 --step-incr 8192
```
