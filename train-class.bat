@REM python .\train-class.py --accelerator "gpu" --exp-name tig-sent --max_epochs 10 --batch-size 128 --hidden-dim 128 --fc-dropout 0.6 --rnn-dropout 0.6 --emb-dropout 0.6 --embedding-dim 300 --dataset-folder dataset/tig --charset-path data/am-charset.txt --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file dataset/corpus/tig/tig-ft.vec --emb-type "emb" --train-model

@REM for DistillEmb Based model
@REM python .\train-class.py --accelerator "gpu" --exp-name tig-sent --max_epochs 10 --batch-size 128 --hidden-dim 128 --fc-dropout 0.3 --rnn-dropout 0.3 --emb-dropout 0.05 --embedding-dim 300 --dataset-folder dataset/tig --charset-path data/am-charset.txt --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file dataset/corpus/tig/tig-ft.vec --emb-type "CNN"

@REM for %%t in (1 2 3 4 5) do (

@REM     python .\train-class.py --accelerator "gpu" --exp-name tig-sent-emb --max_epochs 20 --batch-size 128 --hidden-dim 128 --fc-dropout 0.6 --rnn-dropout 0.6 --emb-dropout 0.6 --embedding-dim 300 --dataset-folder dataset/tig --charset-path data/am-charset.txt --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file dataset/corpus/tig/am-tig-task-ft.vec --emb-type "emb" --trail-id %%t --train-model
@REM )

@REM python .\train-class.py --accelerator "gpu" --exp-name tig-sent-emb --max_epochs 20 --batch-size 128 --hidden-dim 128 --fc-dropout 0.6 --rnn-dropout 0.6 --emb-dropout 0.6 --embedding-dim 300 --dataset-folder dataset/tig --charset-path data/am-charset.txt --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file dataset/corpus/tig/tig-ft.vec --emb-type "emb"  --no-train-model  --test-trail-ids 1-2-3-4-5


@REM for %%t in (1 2 3 4 5) do (

@REM     python .\train-class.py --accelerator "gpu" --exp-name tig-sent-emb-am --max_epochs 20 --batch-size 128 --hidden-dim 128 --fc-dropout 0.6 --rnn-dropout 0.6 --emb-dropout 0.6 --embedding-dim 300 --dataset-folder dataset/tig --charset-path data/am-charset.txt --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file dataset/corpus/tig/am-tig-task-ft.vec --emb-type "emb" --trail-id %%t --train-model
@REM )

@REM python .\train-class.py --accelerator "gpu" --exp-name tig-sent-emb-am --max_epochs 20 --batch-size 128 --hidden-dim 128 --fc-dropout 0.6 --rnn-dropout 0.6 --emb-dropout 0.6 --embedding-dim 300 --dataset-folder dataset/tig --charset-path data/am-charset.txt --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file dataset/corpus/tig/tig-ft.vec --emb-type "emb"  --no-train-model  --test-trail-ids 1-2-3-4-5 --vocab-file "dataset/tig/tig-task-vocab.txt"

