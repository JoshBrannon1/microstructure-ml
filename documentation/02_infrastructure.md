# Microstructure ML — Infrastructure & Deployment

---

## Development Environments

### Original: WSL2 Ubuntu on Windows

| Tool | Detail |
|---|---|
| OS | WSL2 Ubuntu on Windows 11 |
| Shell | ZSH |
| Editor | VS Code (always opened via `code .` from WSL terminal — verify "WSL: Ubuntu" in bottom-left) |
| Package manager | Poetry 2.3.3 via `python3 -m poetry` |
| Venv path | `.venv/Scripts/python.exe` (Windows-style path — WSL creates Windows-compatible venv layouts) |
| Run tests | `make test` → `.venv/Scripts/python.exe -m pytest tests/ -v` |

**Path friction:** WSL creates venv with `Scripts/python.exe` (Windows-style) rather than `bin/python` (Linux-style). This caused friction with any tool expecting Linux layout. Makefile had to use the Windows-style path explicitly.

**Critical rule:** Always work from `~/microstructure-ml` in WSL (not `/mnt/c/...` which is the Windows mount path). Open VS Code with `code .` from WSL terminal to ensure the editor sees the Linux filesystem.

### Current: Native Fedora Linux

Migrated from WSL to native Fedora at a natural checkpoint before running the full pipeline for the first time.

**Why migrate:**
- WSL introduces `Scripts/` vs `bin/` path friction for venv layouts
- WSL path weirdness (`/mnt/c/...` vs `~/`) caused deployment issues earlier in the project
- Native Linux is better long-term for ML work (CUDA, Docker, native tooling)

**Migration checklist executed:**
1. Committed and pushed all changes from WSL
2. On Fedora: `poetry config virtualenvs.in-project true` (ensures venv lands at `.venv/` inside project)
3. Cloned repo, ran `make install`, verified `make test` passes
4. Installed `gcloud` CLI via Google Cloud DNF repo
5. Authenticated with `gcloud init`
6. Transferred data from GCP VM

**Poetry PATH duplication on Fedora:** After installing Poetry, `~/.local/bin` appeared dozens of times in `$PATH`. Root cause: Fedora ZSH config already exports `~/.local/bin`; a manually added export in `~/.zshrc` duplicated it on every shell start. Resolved by removing the duplicate export. Functionally harmless but worth cleaning up.

**Makefile migration:** Changed all Python invocations from `.venv/Scripts/python.exe` to `poetry run python` — which resolves the correct Python regardless of platform, making the Makefile portable across Windows/WSL/Linux.

```makefile
# Before (WSL)
.venv/Scripts/python.exe -m microstructure_ml.collector

# After (cross-platform)
poetry run python -m microstructure_ml.collector
```

---

## GCP Compute Engine VM

### Final Configuration

| Setting | Value |
|---|---|
| Instance name | `microstructure-collector` |
| Machine type | e2-micro |
| Provisioning model | Standard |
| Region/Zone | us-east5-a |
| OS | Debian 12 (Bookworm) |
| Boot disk | 10 GB |
| Auto-restart | Enabled |

### Why Standard (Not Spot) Provisioning

**Spot VMs** are up to 90% cheaper but GCP can preempt them at any time with 30 seconds notice. Critically, **Spot VMs do not support auto-restart** — when preempted, the VM stays off until manually restarted.

**Standard VMs** support GCP's auto-restart policy — GCP brings the VM back up after any interruption. For a research dataset where data continuity matters (gaps = fewer training examples), this is worth the cost.

**Cost:** The e2-micro is within GCP's always-free tier in specified US regions (`us-east1`, `us-west1`, `us-central1`). Free tier covers one standard e2-micro per month. **Note:** us-east5-a is not in the always-free tier — this may incur charges. The always-free tier zones are `us-east1`, `us-west1`, `us-central1`.

**Key lesson:** Provisioning model **cannot be edited** on an existing VM — must delete and recreate. Always choose the correct model at creation time.

### Autonomous Recovery Chain

When everything is configured correctly, recovery is fully automatic:

```
GCP detects VM went down
  → GCP auto-restart policy brings VM back up (standard VMs only)
    → systemd starts on boot
      → systemd starts microstructure-collector service
        → collector connects to Kraken WebSocket
          → data collection resumes
```

No manual intervention required at any step.

---

## GCP Environment Setup

### Python Version Issue

`pyproject.toml` was configured with `python = ">=3.13"` but Debian ships with Python 3.11. The codebase uses no 3.13-specific features, so the fix was to lower the requirement:

```toml
python = ">=3.11"
```

After changing `pyproject.toml`, the lockfile was out of sync:
```bash
poetry lock        # regenerates poetry.lock
poetry install     # installs from updated lock
```

### Poetry Venv Placement

Poetry defaults to placing virtualenvs in `~/.cache/pypoetry/virtualenvs/` on Linux. The Makefile references `.venv/bin/python` (project-local path). Fix:

```bash
poetry config virtualenvs.in-project true
poetry env remove python3
poetry install
```

This creates `.venv/` inside the project directory, matching local development setup.

**Key lesson:** Always set `virtualenvs.in-project true` *before* running `poetry install` on a new machine. If forgotten: remove env and reinstall.

### Hardcoded Path Bug

The collector originally had a hardcoded absolute path:

```python
# Before — broken on any machine that isn't Josh's local setup
write_snapshots(self.buffer, product, exchange, "/home/josh/microstructure-ml/data")
```

This caused a `PermissionError` on the VM. Fix:

```python
# After — relative path works from any machine given correct WorkingDirectory
write_snapshots(self.buffer, product, exchange, "data")
```

**Key lesson:** Hardcoded absolute paths are a deployment hazard. Relative paths work from any machine as long as the working directory is correct — which systemd's `WorkingDirectory` handles.

---

## systemd Service

### Why systemd (not nohup/tmux/screen)

| Tool | Survives SSH disconnect | Auto-restarts on crash | Starts on boot |
|---|---|---|---|
| nohup | yes | no | no |
| tmux/screen | yes | no | no |
| systemd | yes | yes | yes |

systemd is the correct tool because the collector needs to survive crashes and VM reboots without manual intervention.

### Service File

Location: `/etc/systemd/system/microstructure-collector.service`

```ini
[Unit]
Description=Microstructure ML Collector
After=network.target

[Service]
Type=simple
User=brannonjosh11
WorkingDirectory=/home/brannonjosh11/microstructure-ml
ExecStart=/home/brannonjosh11/microstructure-ml/.venv/bin/python -m microstructure_ml.collector
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Key field explanations:**
- `After=network.target` — waits for network before starting; critical for WebSocket connection
- `Restart=on-failure` — restarts only if process exits with a non-zero code
- `RestartSec=10` — waits 10 seconds before restarting; avoids rapid crash loops
- `WorkingDirectory` — sets CWD so relative `data/` path resolves correctly
- `StandardOutput=journal` — routes logs to journald, viewable with `journalctl`

### Useful systemd Commands

```bash
sudo systemctl status microstructure-collector    # check if running
sudo systemctl start microstructure-collector     # start
sudo systemctl stop microstructure-collector      # stop
sudo systemctl restart microstructure-collector   # restart (e.g. after code changes)
sudo systemctl enable microstructure-collector    # auto-start on boot
sudo journalctl -u microstructure-collector -f    # watch live logs
```

### Common Errors

**`status=203/EXEC`:** systemd cannot find or execute the binary at `ExecStart`. Almost always caused by the venv not being in the expected location. Fix: verify `.venv/bin/python` exists; check Poetry venv placement.

**`bad-setting`:** The systemd service file is INI format, not bash. Putting shell commands inside the service file causes this. Rewrite with correct INI format.

### Code Update Workflow

After making changes locally and pushing to GitHub:

```bash
cd ~/microstructure-ml
git pull
sudo systemctl restart microstructure-collector
sudo systemctl status microstructure-collector
```

**Never run `make collect` while the systemd service is active** — two collectors writing to the same `data/` directory simultaneously will cause duplicate snapshots.

---

## Data Transfer: GCP VM to Local Machine

### Standard Transfer (gcloud scp)

```bash
gcloud compute scp --recurse --zone=us-east5-a \
  microstructure-collector:/home/brannonjosh11/microstructure-ml/data/raw/exchange=Kraken/product=BTC-USD \
  data/raw/
```

**Gotcha:** When the destination directory already exists, `scp` nests the source directory *inside* it rather than merging. This produces a duplicated path (`data/raw/exchange=Kraken/exchange=Kraken/...`). Fix: move contents up one level with `mv` and remove the empty duplicate directory.

### Finding Instance Details

```bash
gcloud compute instances list           # shows all instances with name, zone, status
gcloud auth list                        # shows active account
gcloud config list                      # shows current project and zone config
```

---

## Data Recovery: Disk Snapshot Procedure

When the original VM became inaccessible due to a zone capacity issue, data was recovered from GCP disk snapshots.

**Why this works:** Disks are zone-locked (cannot attach a us-east5 disk to a us-east1 VM). Snapshots are global resources and can be used to create disks in any zone.

**Full procedure:**
1. Create a snapshot of the old VM's boot disk in GCP Console → Compute Engine → Disks → Create Snapshot
2. Create a new disk from the snapshot in the target zone (snapshots are global — can create in any zone)
3. Attach the new disk to the new VM as a secondary disk (not the boot disk)
4. Mount the partition inside the new VM:
   ```bash
   lsblk                          # verify partition layout — look for sdb1, not sdb
   sudo mkdir /mnt/olddata
   sudo mount /dev/sdb1 /mnt/olddata   # mount the partition, not the raw disk
   ```
5. Copy the data:
   ```bash
   cp -r /mnt/olddata/home/brannonjosh11/microstructure-ml/data/exchange=Kraken \
     ~/microstructure-ml/data/
   ```

**Key lessons:**
- Always mount the **partition** (`/dev/sdb1`), not the raw disk (`/dev/sdb`). Use `lsblk` to see partition layout.
- The old VM may have multiple home directories — check all of them.
- Snapshots are global; disks are zone-locked.

---

## Storage Details

**Capacity estimate:** ~50–100 MB/day at 1-second cadence, 10 levels deep, 43 columns. A 10 GB boot disk gives roughly 3–6 months of runway before storage becomes a concern.

**Files per day:** 144 files/day (one per 10 minutes). Writing one file per snapshot (86,400/day) would create massive filesystem overhead; 144 files/day is manageable.

**Data verification commands:**

```bash
# Count total files
find data -name '*.parquet' | wc -l

# List most recent files
ls -lt $(find data -name '*.parquet') | head -10

# Inspect a specific file
poetry run python -c "
import polars as pl
df = pl.read_parquet('data/raw/.../part-HH-MM-SS.parquet')
print(df.shape)
print('Min timestamp:', df['timestamp'].min())
print('Max timestamp:', df['timestamp'].max())
print('Bid < Ask always:', (df['bid_price_1'] < df['ask_price_1']).all())
"
```

---

## Issues Encountered

| Issue | Cause | Resolution |
|---|---|---|
| `apt` lock held 20+ minutes on boot | `unattended-upgrades` ran on boot and grabbed dpkg lock | Wait it out; kill the process if it exceeds 20 minutes |
| `dpkg was interrupted` error | Background apt process killed mid-run | `sudo dpkg --configure -a` |
| Poetry can't find compatible Python | `pyproject.toml` required `>=3.13`, VM has 3.11 | Downgrade requirement to `>=3.11`, run `poetry lock && poetry install` |
| Lockfile out of sync after pyproject change | `poetry.lock` generated under old requirements | `poetry lock` to regenerate |
| Venv not in project directory | Poetry defaults to `~/.cache/pypoetry/virtualenvs/` | `poetry config virtualenvs.in-project true`, remove env, reinstall |
| `PermissionError` writing to `/home/josh` | Hardcoded absolute path in `collector.py` | Change to relative path `"data"` |
| `status=203/EXEC` in systemd | `.venv/bin/python` didn't exist | Fix venv placement, verify path |
| `bad-setting` in systemd service | Bash commands pasted into service file (INI format required) | Rewrite as INI |
| Overnight data gaps | Spot VM preempted; no auto-restart support | Switch to standard provisioning |
| Cannot edit provisioning model | GCP doesn't allow changing on existing VMs | Delete and recreate VM as standard |
| Cannot attach old disk to new zone | Disks are zone-locked | Snapshot → create new disk in target zone → attach |
| `mount: wrong fs type` on `/dev/sdb` | Tried to mount raw disk instead of partition | Use `/dev/sdb1` (confirmed with `lsblk`) |
| `git pull` didn't update code | Local uncommitted change conflicted with incoming commit | `git restore <file>` to discard local change, then `git pull` |
| Duplicated path after `gcloud scp` | `scp` nests source inside existing destination dir | Move contents up one level with `mv` |
| Poetry PATH duplication on Fedora | Duplicate `~/.local/bin` exports in ZSH config | Remove duplicate export from `~/.zshrc` |

---

## Makefile Reference

```makefile
make install    → poetry install
make collect    → poetry run python -m microstructure_ml.collector
make features INPUT=data/raw [OUTPUT=data/features]
make labels   INPUT=data/features [OUTPUT=data/labels] [INTERVALS="5 10 30"]
make train    INPUT=data/labels [VAL_DAYS=3] [STEP_DAYS=1] [TEST_DAYS=5] \
              [KEEP_COLUMNS="mid_price spread imbalance microprice depth_imbalance"] \
              [LABEL=return_5s] [MODEL_CLASS=linear]
make test     → poetry run python -m pytest tests/ -v
```

**Variable defaults** use `?=` (Make's conditional assignment — only sets if not already provided by caller). `INPUT` intentionally has no default — wrong data path could silently corrupt results.
