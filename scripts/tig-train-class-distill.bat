cd ../
for %%t in (2) do (

    python .\train-class.py --accelerator "gpu" --exp-name tig-sent-distill-25 --max_epochs 20 --batch-size 128 --hidden-dim 128 --fc-dropout 0.0 --rnn-dropout 0.0 --emb-dropout 0.0 --embedding-dim 300 --dataset-folder dataset/tig --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.0001 --vector-file scratch --emb-type "CNN" --trial-id %%t --train-model --data-size 0.25
)


@REM python .\train-class.py --accelerator "gpu" --exp-name tig-sent-emb --max_epochs 20 --batch-size 128 --hidden-dim 128 --fc-dropout 0.6 --rnn-dropout 0.6 --emb-dropout 0.6 --embedding-dim 300 --dataset-folder dataset/tig --charset-path data/am-charset.txt --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file dataset/corpus/tig/tig-task-ft.vec --emb-type "emb"  --no-train-model  --test-trial-ids 1-2-3-4-5 --vocab-file dataset/tig/tig-task-vocab.txt


