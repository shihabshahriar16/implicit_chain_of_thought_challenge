export EPOCHS=15
export LR=5e-5
export FOLDER=math_scaffolding_formula
export MODEL=EleutherAI/gpt-j-6b
export ACC=1
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_gptj.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --accumulate $ACC \
    > log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.acc$ACC 2>&1&

export EPOCHS=15
export LR=1e-5
export FOLDER=math_scaffolding_formula
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_gptj.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --accumulate $ACC \
    > log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.acc$ACC 2>&1&

export EPOCHS=15
export LR=3e-4
export FOLDER=math_scaffolding_formula
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_gptj.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --accumulate $ACC \
    > log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.acc$ACC 2>&1&

export EPOCHS=15
export LR=1e-4
export FOLDER=math_scaffolding_formula
CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_gptj.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --accumulate $ACC \
    > log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.acc$ACC 2>&1&