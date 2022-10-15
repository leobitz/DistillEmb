cd ../
for %%t in (1) do (

<<<<<<<< HEAD:scripts/tig-train-class-distill.bat
    python .\train-class.py --accelerator "gpu" --exp-name tig-sent-distill-25 --max_epochs 60 --batch-size 128 --hidden-dim 256 --fc-dropout 0.0 --rnn-dropout 0.0 --emb-dropout 0.05 --embedding-dim 300 --dataset-folder dataset/tig --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file scratch --emb-type "CNN" --trial-id %%t --train-model --data-size 0.25 
========
    python .\train-class.py --accelerator "gpu" --exp-name tig-sent-emb-am-50 --max_epochs 20 --batch-size 128 --hidden-dim 128 --fc-dropout 0.6 --rnn-dropout 0.6 --emb-dropout 0.6 --embedding-dim 300 --dataset-folder dataset/tig --charset-path data/am-charset.txt --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file dataset/corpus/tig/am-tig-task-ft.vec --emb-type "emb" --trial-id %%t --train-model --vocab-file dataset/tig/tig-task-vocab.txt --data-size 0.50

    python .\train-class.py --accelerator "gpu" --exp-name tig-sent-emb-am-25 --max_epochs 20 --batch-size 128 --hidden-dim 128 --fc-dropout 0.6 --rnn-dropout 0.6 --emb-dropout 0.6 --embedding-dim 300 --dataset-folder dataset/tig --charset-path data/am-charset.txt --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file dataset/corpus/tig/am-tig-task-ft.vec --emb-type "emb" --trial-id %%t --train-model --vocab-file dataset/tig/tig-task-vocab.txt --data-size 0.25
>>>>>>>> 38996721dca20631e640a9d81f724cc483557b37:scripts/tig-train-class.bat
)


@REM python .\train-class.py --accelerator "gpu" --exp-name tig-sent-emb --max_epochs 20 --batch-size 128 --hidden-dim 128 --fc-dropout 0.6 --rnn-dropout 0.6 --emb-dropout 0.6 --embedding-dim 300 --dataset-folder dataset/tig --charset-path data/am-charset.txt --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file dataset/corpus/tig/tig-task-ft.vec --emb-type "emb"  --no-train-model  --test-trial-ids 1-2-3-4-5 --vocab-file dataset/tig/tig-task-vocab.txt


