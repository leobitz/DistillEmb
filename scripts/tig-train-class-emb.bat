cd ../
for %%t in (1 2) do (

    @REM python .\train-class.py --accelerator "gpu" --exp-name tig-sent-emb-100 --max_epochs 30 --batch-size 128 --hidden-dim 256 --fc-dropout 0.6 --rnn-dropout 0.6 --emb-dropout 0.6 --embedding-dim 300 --dataset-folder dataset/tig --charset-path data/am-charset.txt --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file dataset/corpus/tig/tig-task-ft.vec --emb-type "emb" --trial-id %%t --train-model --vocab-file dataset/tig/tig-task-vocab.txt --data-size 1.0

    @REM python .\train-class.py --accelerator "gpu" --exp-name tig-sent-emb-75 --max_epochs 30 --batch-size 128 --hidden-dim 256 --fc-dropout 0.6 --rnn-dropout 0.6 --emb-dropout 0.6 --embedding-dim 300 --dataset-folder dataset/tig --charset-path data/am-charset.txt --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file dataset/corpus/tig/tig-task-ft.vec --emb-type "emb" --trial-id %%t --train-model --vocab-file dataset/tig/tig-task-vocab.txt --data-size 0.75

    @REM python .\train-class.py --accelerator "gpu" --exp-name tig-sent-emb-50 --max_epochs 30 --batch-size 128 --hidden-dim 256 --fc-dropout 0.6 --rnn-dropout 0.6 --emb-dropout 0.6 --embedding-dim 300 --dataset-folder dataset/tig --charset-path data/am-charset.txt --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file dataset/corpus/tig/tig-task-ft.vec --emb-type "emb" --trial-id %%t --train-model --vocab-file dataset/tig/tig-task-vocab.txt --data-size 0.50

    @REM python .\train-class.py --accelerator "gpu" --exp-name tig-sent-emb-25 --max_epochs 30 --batch-size 128 --hidden-dim 256 --fc-dropout 0.6 --rnn-dropout 0.6 --emb-dropout 0.6 --embedding-dim 300 --dataset-folder dataset/tig --charset-path data/am-charset.txt --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file dataset/corpus/tig/tig-task-ft.vec --emb-type "emb" --trial-id %%t --train-model --vocab-file dataset/tig/tig-task-vocab.txt --data-size 0.25
@REM )

 python .\train-class.py --accelerator "gpu" --exp-name tig-sent-emb-25 --max_epochs 30 --batch-size 128 --hidden-dim 256 --fc-dropout 0.6 --rnn-dropout 0.6 --emb-dropout 0.6 --embedding-dim 300 --dataset-folder dataset/tig --charset-path data/am-charset.txt --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file dataset/corpus/tig/tig-task-ft.vec --emb-type "emb" --vocab-file dataset/tig/tig-task-vocab.txt --data-size 0.25 --no-train-model --test-trial-ids 1-2-3-4-5 
