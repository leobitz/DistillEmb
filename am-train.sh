# for r in 1
# do
#     python train-class.py --accelerator "gpu" --exp-name 'am-sent-distill' --max_epochs 60 --batch-size 64 --hidden-dim 256 --fc-dropout 0.0 --rnn-dropout 0.0 --emb-dropout 0.00 --embedding-dim 300 --dataset-folder 'dataset/am/sent' --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file "saves/epoch=91-val_loss=0.00000-val_f1=0.00000.ckpt" --emb-type "CNN" --trial-id 1 --train-model --data-size 1.0 --step-gamma .90  --model-size "small" --grad-accumulate 1
# done
for r in 1
do
    python train-class.py --accelerator "gpu" --exp-name 'tig-sent' --max_epochs 60 --batch-size 128 --hidden-dim 256 --fc-dropout 0.1 --rnn-dropout 0.1 --emb-dropout 0.1 --embedding-dim 300 --dataset-folder 'dataset/tig' --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file "saves/epoch=31-val_loss=0.00000-val_f1=0.00000.ckpt" --emb-type "CNN" --trial-id 1 --train-model --data-size 1.0 --step-gamma .94  --model-size "small" --grad-accumulate 1
done
