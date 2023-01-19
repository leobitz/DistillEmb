for r in 1
do

    # python train-seq.py  --emb-type word --charset-path data/am-charset.txt  --dataset-folder ./dataset/am/ner --max-seq-len 200 --batch-size 64 --learning-rate 0.001 --embedding-dim 300 --hidden-dim 256 --num-rnn-layers 1 --rnn-type LSTM --fc-dropout 0.6 --rnn-dropout 0.6 --emb-dropout 0.6  --max_epochs 60  --emb-type "CNN" --trial-id $r --train-model --vector-file scratch

    python .\train-class.py --accelerator "gpu" --exp-name am-sent-distill --max_epochs 60 --batch-size 128 --hidden-dim 256 --fc-dropout 0.3 --rnn-dropout 0.3 --emb-dropout 0.1 --embedding-dim 300 --dataset-folder dataset/tig --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file saves/new-tig-distill-tig-cnn768-w1/epoch=127-val_loss=0.00000-val_f1=0.00000.ckpt --emb-type "CNN" --trial-id %%t --train-model --data-size 0.25 --step-gamma .96 --grad-accumulate 4 --model-size large

done
