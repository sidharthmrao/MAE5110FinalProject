# Phase 1 — Go2 Force-Based Policy Deployment (Linux)

## 1. Setup

From the repository root:

```bash
git switch dev
git pull --ff-only origin dev
uv sync --frozen
```

## 2. Run the policy in simulation

```bash
uv run --frozen python scripts/play.py \
  --task go2osc \
  --device cpu \
  --num_envs 1 \
  --experiment_name go2osc_fixed_orientation \
  --load_run Nov16_18-11-18_ \
  --checkpoint 500 \
  --original_cfg \
  --no-keyboard
```

The original training configuration must be available in
`logs/go2osc_fixed_orientation/Nov16_18-11-18_/files/`.

If it is missing, obtain the saved training configuration
from the original training run before proceeding.

## 3. Verify the deployment configuration

```bash
uv run --frozen pytest tests/unit_tests/test_go2osc_deploy.py -q
```

```bash
GO2_PHASE=phase1 uv run --frozen python -c "from go2_deploy.deploy_config import DeployConfig; c=DeployConfig(); print(c.task_name, c.obs_dim(), c.ctrl_freq)"
```

Expected output: `go2osc 65 50`

## 4. Export the policy

```bash
GO2_PHASE=phase1 uv run --frozen python \
  scripts/export_go2_policy.py \
  --name phase1_dev \
  --experiment_name go2osc_fixed_orientation \
  --run_name Nov16_18-11-18_ \
  --checkpoint 500
```

The exported policy is saved in `logs/policies/phase1_dev/`.

## 5. Verify the exported policy offline

```bash
GO2_PHASE=phase1 uv run --frozen python -c "from go2_deploy.policy_bundle import load_bundle; p,c=load_bundle('logs/policies/phase1_dev'); print(c.task_name,c.obs_dim())"
```

Expected output: `go2osc 65`

## 6. Deploy on the real robot

Follow `README_DEPLOY.md` for Unitree SDK installation
and Ethernet configuration.

Before deployment, verify the force sensor calibration,
joint position limits and emergency-stop procedure with course staff.

Once approved, run:

```bash
GO2_PHASE=phase1 uv run --frozen --extra unitree_sdk \
  python -m go2_deploy.deploy eth0 \
  --bundle logs/policies/phase1_dev
```

Replace `eth0` with the appropriate network interface.

## Policy configuration

- Observation dimension: 65
- Action dimension: 12
- Control frequency: 50 Hz
- Nominal oscillator frequency: 2 Hz
- Oscillator frequency modulation: GRF feedback
- Target GRF observation: enabled
- Checkpoint: 500
- Branch: dev

## Important

The dev and dev-sid policies have different observation
dimensions and oscillator dynamics. Always use the corresponding
branch and policy bundle.

Physical deployment remains subject to hardware calibration
and supervised testing.