for r in 1
do

    python train-class.py --accelerator "gpu" --exp-name 'am-sent-distill' --max_epochs 60 --batch-size 128 --hidden-dim 64 --fc-dropout 0.3 --rnn-dropout 0.3 --emb-dropout 0.1 --embedding-dim 300 --dataset-folder 'dataset/am/sent' --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file "saves/distill-am/epoch=47-val_loss=0.00000-val_f1=0.00000.ckpt" --emb-type "CNN" --trial-id $r --train-model --data-size 1.0 --step-gamma .96  --model-size large --grad-accumulate 1
done
