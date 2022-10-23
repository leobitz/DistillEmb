cd ../
for %%t in (1 2) do (

python .\train-class.py --accelerator "gpu" --exp-name tig-sent-distill-25-256 --max_epochs 60 --batch-size 64 --hidden-dim 256 --fc-dropout 0.2 --rnn-dropout 0.2 --emb-dropout 0.1 --embedding-dim 300 --dataset-folder dataset/tig --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file logs/tig-distill-full32/version_0/checkpoints/epoch=63-step=316288.ckpt --emb-type "CNN" --trial-id %%t --train-model --data-size .25 --step-gamma .9

@REM python .\train-class.py --accelerator "gpu" --exp-name tig-sent-distill-25-256 --max_epochs 60 --batch-size 64 --hidden-dim 256 --fc-dropout 0.2 --rnn-dropout 0.2 --emb-dropout 0.05 --embedding-dim 300 --dataset-folder dataset/tig --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file logs/tig-distill-full64/version_0/checkpoints/epoch=63-step=316288.ckpt --emb-type "CNN" --trial-id %%t --train-model --data-size 0.25 --step-gamma .9

)

@REM python .\train-class.py --accelerator "gpu" --exp-name tig-sent-distill-100 --max_epochs 60 --batch-size 128 --hidden-dim 128 --fc-dropout 0.6 --rnn-dropout 0.4 --emb-dropout 0.1 --embedding-dim 300 --dataset-folder dataset/tig --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file logs/tig-distill-full/version_1/checkpoints/epoch=7-step=39536.ckpt --emb-type "CNN"  --data-size 1.0 --step-gamma .9 --no-train-model  --test-trial-ids 1-2


@REM python .\train-class.py --accelerator "gpu" --exp-name tig-sent-emb --max_epochs 20 --batch-size 128 --hidden-dim 128 --fc-dropout 0.6 --rnn-dropout 0.6 --emb-dropout 0.6 --embedding-dim 300 --dataset-folder dataset/tig --charset-path data/am-charset.txt --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file dataset/corpus/tig/tig-task-ft.vec --emb-type "emb"  --no-train-model  --test-trial-ids 1-2-3-4-5 --vocab-file dataset/tig/tig-task-vocab.txt


