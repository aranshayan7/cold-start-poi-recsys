#!/usr/bin/env bash
# Run the experiment grid of the paper.
#
#   scripts/run_experiments.sh quick   # sample data, one seed, 2 epochs (20 to 30 minutes on a laptop CPU)
#   scripts/run_experiments.sh full    # full data, 3 seeds, all variants (hours)
#
# Environment variables:
#   DATA     directory with the processed data (default: data/sample for quick, data/processed for full)
#   OUT      directory for results            (default: results/<profile>)
#   DEVICE   cpu | cuda | mps                  (default: cpu)
set -euo pipefail

PROFILE="${1:-quick}"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
DEVICE="${DEVICE:-cpu}"

case "$PROFILE" in
  quick) DATA="${DATA:-data/sample}";    OUT="${OUT:-results/quick}"; SEEDS="1";     EXTRA="--epochs 2" ;;
  full)  DATA="${DATA:-data/processed}"; OUT="${OUT:-results/full}";  SEEDS="1 2 3"; EXTRA="" ;;
  *) echo "usage: $0 {quick|full}" >&2; exit 2 ;;
esac

run() { python -m csrec --data "$DATA" --out "$OUT" --device "$DEVICE" $EXTRA "$@"; }

# 1. Non-learned references (no training).
for s in $SEEDS; do
  run --learner pop      --seed "$s"
  run --learner localpop --seed "$s"
done

# 2. Main comparison: the two meta-learners on the same shared representation.
for s in $SEEDS; do
  run --learner proto --seed "$s"
  run --learner maml  --seed "$s"
done

# 3. Prototypical learner with the popularity feature (best configuration).
for s in $SEEDS; do
  run --learner proto --ablation pop_feat --seed "$s"
done

# 4. Ablations of the shared representation (seed 1 is enough for a first look).
for abl in no_geo no_social no_id; do
  run --learner proto --ablation "$abl" --seed 1
done

if [ "$PROFILE" = "full" ]; then
  # 5. Support-size sweep and mixed (hard) training negatives.
  for s in $SEEDS; do
    for k in 1 3 5; do
      run --learner proto --ablation pop_feat --k "$k" --seed "$s"
      run --learner pop --k "$k" --seed "$s"
    done
  done
  run --learner proto --ablation pop_feat --train_neg mixed --seed 1
  run --learner maml  --alpha_in 0.01 --seed 1
fi

python scripts/aggregate.py --results "$OUT"
