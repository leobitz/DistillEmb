cd ../
for %%t in (1) do (

    python train-seq.py  --emb-type word --charset-path data/am-charset.txt  --dataset-folder ./dataset/am/ner --max-seq-len 200 --batch-size 64 --learning-rate 0.001 --embedding-dim 300 --hidden-dim 256 --num-rnn-layers 1 --rnn-type LSTM --fc-dropout 0.6 --rnn-dropout 0.6 --emb-dropout 0.6  --max_epochs 60 --vector-file dataset/corpus/vectors/am-ft.vec --emb-type "emb" --trial-id %%t --train-model
)