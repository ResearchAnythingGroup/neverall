# Review for "A Durable Unlearning Enhancement Framework to Nullify Recall of Sensitive Data on Incremental Training"

## 1. DUE Framework implementation
train_test_utils.py: Implementation of the DUE post-MU training framework
/nets/forget_protonet.py: Implementation of the prototype-based contrastive learning algorithm

## 2. Implementation of Machine Unlearning Methods
/unlearn: The directory accommodates all implementations of MU methods used in the paper.

## 3. Scripts for Experiments
### 3.1 Script Directory Structure
/scripts: The directory accommodates all scripts for the experiments.
- /scripts/cifar10: scripts for CIFAR-10
- /scripts/stl10: scripts for STL-10
- /scripts/flower102: scripts for Flower-102
- /scripts/pet37: scripts for Pet-37

### 3.2 Script Directory Structure
Under each script directory for a dataset, e.g. /scripts/cifar10 for CIFAR-10, it contains
1. gendata.sh: generate train and test data for experiments
2. train.sh: training model with $D_{tr}$ for TRM 
3. unlearn.sh: running all the MU methods with $D_{f}$  for ULM
4. finetune_neverecall.sh: running the full DUE framework
5. finetune_ssgs.sh: running DUE with the ablation of SSGS compnent
6. finetune_sspr: running DUE with the ablation of SSPR compnent
7. evals.sh: evaluating the experimental results
8. visuals.sh: draw figures for the experiments