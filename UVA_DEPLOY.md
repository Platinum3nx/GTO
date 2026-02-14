# 🚀 How to Run GTO on UVA Cluster

Follow these steps exactly to run your training on the UVA Computer Science cluster.

## 1. Connect to UVA Network
If you are **off-campus**, you must connect to the **UVA High Security VPN** (or standard VPN) first.
- Download/Open **Cisco Secure Client**.
- Connect to `vpn.virginia.edu`.

## 2. Open Terminal
Open your terminal (PowerShell on Windows, Terminal on Mac).

## 3. Transfer Your Code
You need to copy your `GTO` folder from your laptop to the cluster. Run this command on your laptop's terminal:

```bash
# Replace /Users/arjunmalghan/GTO with the actual path to your folder
rsync -avz --exclude '.git' --exclude 'venv' --exclude 'runs' /Users/arjunmalghan/GTO qat5sc@portal.cs.virginia.edu:~/
```

*It will ask for your CS password.*

## 4. SSH into the Cluster
Log in to the cluster:

```bash
ssh qat5sc@portal.cs.virginia.edu
```

*Enter your password again.*

## 5. Submit the Job
Once you are logged in, run these commands:

```bash
# Go to the folder you just uploaded
cd GTO

# Submit the training job to the queue
sbatch run_uva.slurm
```

## 6. Monitor Progress

**Check if it's running:**
```bash
squeue -u qat5sc
```
*If you see your job listed with `Running` or `R`, it's working!*

**View Live Logs:**
The script creates output files in the `runs/` folder.
```bash
# List log files (e.g., slurm-12345.out)
ls -l runs/

# Read the latest log (adjust the number to match yours)
tail -f runs/slurm-*.out
```
*(Press `Ctrl+C` to stop watching the log)*

---

### ⚡️ Quick Resume
If you disconnect, the job **keeps running**. You can log back in later and check `squeue` or the log files to see progress.
