cd ../
for %%t in (1) do (

python train-class.py --accelerator "gpu" --exp-name am-sent-distill --max_epochs 60 --batch-size 32 --hidden-dim 64 --fc-dropout 0.3 --rnn-dropout 0.3 --emb-dropout 0.1 --embedding-dim 300 --dataset-folder dataset/am/sent --charset-path "data/am-charset.txt" --max-seq-len 20 --num-rnn-layers 1 --learning-rate 0.001 --vector-file "scratch" --emb-type "CNN" --trial-id 1 --train-model --data-size 0.25 --step-gamma .96  --model-size large --grad-accumulate 1


@REM python .\train-class.py --accelerator "gpu" --exp-name x-tig-sent-distill-25cnn768-127e --max_epochs 180 --batch-size 32 --hidden-dim 256 --fc-dropout 0.3 --rnn-dropout 0.3 --emb-dropout 0.1 --embedding-dim 300 --dataset-folder dataset/tig --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file saves/new-tig-distill-tig-cnn768/epoch=127-val_loss=0.00000-val_f1=0.00000.ckpt --emb-type "CNN" --trial-id %%t --train-model --data-size 0.25 --step-gamma .96 --grad-accumulate 4


@REM python .\train-class.py --accelerator "gpu" --exp-name xw1-tig-sent-distill-25cnn768-63e --max_epochs 180 --batch-size 32 --hidden-dim 256 --fc-dropout 0.3 --rnn-dropout 0.3 --emb-dropout 0.1 --embedding-dim 300 --dataset-folder dataset/tig --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file saves/new-tig-distill-tig-cnn768-w1/epoch=63-val_loss=0.00000-val_f1=0.00000.ckpt --emb-type "CNN" --trial-id %%t --train-model --data-size 0.25 --step-gamma .96 --grad-accumulate 4


@REM python .\train-class.py --accelerator "gpu" --exp-name x-tig-sent-distill-25cnn768-63e --max_epochs 180 --batch-size 32 --hidden-dim 256 --fc-dropout 0.3 --rnn-dropout 0.3 --emb-dropout 0.1 --embedding-dim 300 --dataset-folder dataset/tig --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file saves/new-tig-distill-tig-cnn768/epoch=63-val_loss=0.00000-val_f1=0.00000.ckpt --emb-type "CNN" --trial-id %%t --train-model --data-size 0.25 --step-gamma .96 --grad-accumulate 4


@REM python .\train-class.py --accelerator "gpu" --exp-name xw1-tig-sent-distill-25 --max_epochs 60 --batch-size 64 --hidden-dim 256 --fc-dropout 0.3 --rnn-dropout 0.3 --emb-dropout 0.1 --embedding-dim 300 --dataset-folder dataset/tig --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file saves/new-tig-distill-tig-w1/epoch=63-val_loss=0.00000-val_f1=0.00000.ckpt --emb-type "CNN" --trial-id %%t --train-model --data-size 0.25 --step-gamma .96 --grad-accumulate 2


@REM python .\train-class.py --accelerator "gpu" --exp-name x-tig-sent-distill-25 --max_epochs 60 --batch-size 64 --hidden-dim 256 --fc-dropout 0.3 --rnn-dropout 0.3 --emb-dropout 0.1 --embedding-dim 300 --dataset-folder dataset/tig --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file saves/new-tig-distill-tig-w1/epoch=63-val_loss=0.00000-val_f1=0.00000.ckpt --emb-type "CNN" --trial-id %%t --train-model --data-size 0.25 --step-gamma .96 --grad-accumulate 2

)


@REM python .\train-class.py --accelerator "gpu" --exp-name tig-sent-emb --max_epochs 20 --batch-size 128 --hidden-dim 128 --fc-dropout 0.6 --rnn-dropout 0.6 --emb-dropout 0.6 --embedding-dim 300 --dataset-folder dataset/tig --charset-path data/am-charset.txt --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file dataset/corpus/tig/tig-task-ft.vec --emb-type "emb"  --no-train-model  --test-trial-ids 1-2-3-4-5 --vocab-file dataset/tig/tig-task-vocab.txt


