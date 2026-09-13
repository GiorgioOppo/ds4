# DeepSeek V4.1: two-machine ROCm cluster

- Two ROCm/gfx1151 machines; tested with 128 GB RAM each.
- Same engine revision and `DeepSeek-V4.1-Flash-Q2.gguf` on both. Each keeps the full GGUF; resident weights are approximately 80.6 GiB per rank. Engram stays on disk.
- Exactly one coordinator and one worker. Attention is tensor-parallel; routed MoE is expert-parallel (192 whole experts per rank). Both execute every layer; KV and the output head are replicated. No `--layers`, SSD expert streaming or DSpark.
- All three transports require a reachable TCP control address. Use a trusted network: peer traffic has no authentication or encryption.
- Build both peers with `make strix-halo ROCM_ARCH=gfx1151` after installing any required RoCE headers.
- Run from the engine build directory. Set these variables in **both** terminals; `MODEL` may differ between machines:

```bash
MODEL=/absolute/path/DeepSeek-V4.1-Flash-Q2.gguf
COORD=10.99.0.1       # Coordinator's address on the selected link
CTX=16384
```

## TCP over Ethernet

- Working Ethernet/IP connection; coordinator TCP port 9911 reachable from the worker.
- No verbs packages or USB stream device required.

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

## USB4STREAM

- USB4/Thunderbolt host-to-host cable and kernel with `CONFIG_USB4_STREAM` and `CONFIG_USB4_CONFIGFS` (tested: Linux 7.2.5).
- Keep IP connectivity for control; USB Ethernet over the same cable is sufficient. Set `COORD` to its coordinator address.
- If the cable is the only IP path, assign unused addresses to its USB Ethernet interface first; do not replace a management/default route:

```bash
USB_IF=thunderbolt0            # Use the USB Ethernet name printed by ip -br link
# Coordinator only
sudo ip link set "$USB_IF" up
sudo ip address add 10.99.0.1/30 dev "$USB_IF"
# Worker only
sudo ip link set "$USB_IF" up
sudo ip address add 10.99.0.2/30 dev "$USB_IF"
# Both terminals: COORD=10.99.0.1
```

- Create **one bidirectional stream** on each host. The stream name must match; device indexes can differ.
- Setup runs on the hosts as root. Inference runs as your ordinary user.

```bash
# Worker first, then coordinator after the worker HopID allocation below
sudo modprobe thunderbolt_net
sudo modprobe thunderbolt_stream
mountpoint -q /sys/kernel/config || sudo mount -t configfs none /sys/kernel/config
ip -br address                  # Identify the USB Ethernet interface/address

# On the host being configured: identify the connected peer's stream service; never guess its number.
for key in /sys/bus/thunderbolt/devices/*/key; do
  [ "$(cat "$key")" = stream ] && dirname "$key"
done
SERVICE=1-2.0                  # Replace with that host's printed service name
STREAM=/sys/kernel/config/thunderbolt/stream/$SERVICE/ds4data
sudo mkdir -p "$(dirname "$STREAM")"
sudo mkdir "$STREAM"           # Refuses to overwrite an existing stream
```

Run the preceding setup on the **worker first**, then allocate its HopIDs:

```bash
# Worker only; complete this before creating the coordinator's ds4data directory.
printf '%s\n' -1 | sudo tee "$STREAM/in_hopid" "$STREAM/out_hopid"
```

Now run the setup block on the coordinator. Its same-named stream adopts the worker's advertised HopIDs. On **both** hosts:

```bash
cat "$STREAM/in_hopid" "$STREAM/out_hopid"   # Both >= 8; coordinator IN = worker OUT and vice versa
USB_DEV=/dev/tbstream$(cat "$STREAM/index")
sudo udevadm settle
sudo chown "$(id -u):$(id -g)" "$USB_DEV"
sudo chmod 600 "$USB_DEV"
test -c "$USB_DEV" && test -r "$USB_DEV" && test -w "$USB_DEV"
```

```bash
# Coordinator
./ds4-server --rocm -m "$MODEL" --ctx "$CTX" \
  --tensor-parallel --role coordinator --listen "$COORD" 9911 \
  --transport usb4stream --usb4stream-device "$USB_DEV" \
  --batched-session 1 --host 127.0.0.1 --port 8080

# Worker
./ds4 --rocm -m "$MODEL" --ctx "$CTX" \
  --tensor-parallel --role worker --coordinator "$COORD" 9911 \
  --transport usb4stream --usb4stream-device "$USB_DEV"
```

- Startup must report `transport=usb4stream` and the intended device.
- Configuration and permissions are temporary; recreate after reboot/reconnection if lost. Stop inference before removing your stream with `sudo rmdir "$STREAM"`.
- Containers need the configured character device, GPU devices and TCP connectivity. Kernel/module setup belongs on the host.

### Tested Strix Halo interrupt fix

- Stock 7.2.5 stalled during sustained stream traffic on the tested AMD controller.
- Applied one change in `ring_clear_msix()` in `drivers/thunderbolt/nhi.c`: read back the interrupt register after its posted clear write.
- Based on Jonathan Yates's [MSI-X clear patch](https://github.com/jyatesdotdev/strix-rdma/blob/d19af99ce91abda691a2bd0f21eb114e0a64bacd/kernel/zerocopy/0011-thunderbolt-Flush-posted-MSI-X-interrupt-clears.patch). The separate RX-prime patch was **not** applied.
- Tested by hot-loading a matching `thunderbolt.ko`; no reboot or persistent boot/module installation. Replacing it interrupts every Thunderbolt user, including USB Ethernet.
- Build against the running distribution kernel's matching source, configuration and headers. Do not load the test binary into another kernel. Secure Boot may require module signing.
- Patch, build, temporary load and rollback: [USB4STREAM kernel fix](USB4STREAM_KERNEL.md). TCP and RoCE do not require this patch.

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

- Linux `rdma` uses RoCE RC with registered host staging; no RCCL or GPUDirect requirement.
- Explicit `tcp`, `usb4stream` or `rdma` fails if unavailable. `auto` negotiates configured RoCE, then configured USB4STREAM, then TCP at connection setup; no mid-generation fallback.

## Vision and first request

- Add `--vision /absolute/path/DeepSeek-V4.1-Flash-Vision.gguf` to **both** commands. Keep `--ctx` equal on both.
- The HTTP API runs only on the coordinator. Test port 8080 after startup; do not send HTTP to peer port 9911.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Say hello."}],"temperature":0,"max_tokens":64,"thinking":false}'
```


## Measured performance

Two 128 GB Strix Halo machines, ROCm 10.0, Q2 weights, context allocation 69,632, greedy generation of 128 tokens (127 steady), no DSpark or images. Values are **prefill / decode tokens/s**.

| Populated context | TCP, 100 GbE | USB4STREAM, 40 Gb/s | RoCE RC, 100 GbE |
|---|---:|---:|---:|
| 1,024: 768-token append | 72.29 / 6.92 | 69.54 / 7.08 | 71.62 / 6.97 |
| 8,192: full prefix | 247.59 / 6.69 | 232.39 / 6.72 | 247.53 / 6.74 |
| 16,384: full prefix | 259.44 / 6.35 | 244.31 / 6.39 | 259.07 / 6.40 |
| 65,536: full prefix | 228.31 / 5.07 | 215.83 / 5.06 | 228.01 / 5.12 |

- Native `ds4-bench` measurements; one matched run per cell. The 1K case excludes its initial 256-token frontier. The deeper cases use the native timing loop with an excluded 256-token/128-output warmup, then a fresh session before measuring the complete prefix. The warmup adapter does not change engine objects or the measured loop.
- Both peers passed continuous fan/profile readiness checks. All frontier logits and printed continuations matched across transports at each depth. No OOM; minimum usable RAM 34.1 GiB. Host swap activity and some sampled process swap were nonzero.
- USB4STREAM used the temporary interrupt-readback patch described above. These physical-link/configuration results are not a universal protocol ranking.
- V4.1 CED activates about 8B parameters/token during prefill and 16B during decode. Full long prefixes reach the decoder-suffix optimization; short appends may not. [Model architecture](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/README.md?code=true).
- Short decode traces show roughly 86 ms/token in local kernels and 22 ms/token in guarded reductions on each peer. RoCE reduces coordinator wait from roughly 26 to 22 ms/token, but this wait also includes peer readiness and CPU scheduling. Faster networking alone does not double decode throughput. Traces include profiler overhead and are separate from the table.
- The 64K trace confirms CED: all 20 encoder layers process 65,536 tokens; decoder work totals only 24,150 layer-rows. Local kernels account for 79–83% of the traced prefill window.
- Deep decode slows mainly in indexer scoring: about 1 ms/token at 1K grows to 52–54 ms/token at 64K, while main attention remains about 5 ms and network wait does not grow. Indexer kernels and guarded-reduction launch sizing are the first optimization targets; no speedup from those changes is claimed here.
