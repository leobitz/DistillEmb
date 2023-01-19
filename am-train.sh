for r in 1 2 3 4 5
do
    python train-class.py --accelerator "gpu" --exp-name 'am-sent-distill' --max_epochs 30 --batch-size 128 --hidden-dim 256 --fc-dropout 0.0 --rnn-dropout 0.0 --emb-dropout 0.00 --embedding-dim 300 --dataset-folder 'dataset/am/sent' --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file "saves/epoch=91-val_loss=0.00000-val_f1=0.00000.ckpt" --emb-type "CNN" --trial-id 1 --train-model --data-size 1.0 --step-gamma .90  --model-size "small" --grad-accumulate 1
done
