# for r in 1 2 3 4 5
# do

# python ./train-class.py --accelerator "gpu" --exp-name tig-sent-scratch-25 --max_epochs 60 --batch-size 128 --hidden-dim 256 --fc-dropout 0.2 --rnn-dropout 0.2 --emb-dropout 0.05 --embedding-dim 300 --dataset-folder dataset/tig --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file scratch --emb-type "CNN" --trial-id $r --train-model --data-size .25 --step-gamma .94


# python ./train-class.py --accelerator "gpu" --exp-name tig-sent-scratch-50 --max_epochs 60 --batch-size 128 --hidden-dim 256 --fc-dropout 0.2 --rnn-dropout 0.2 --emb-dropout 0.05 --embedding-dim 300 --dataset-folder dataset/tig --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file scratch --emb-type "CNN" --trial-id $r --train-model --data-size .50 --step-gamma .94


# python ./train-class.py --accelerator "gpu" --exp-name tig-sent-scratch-75 --max_epochs 60 --batch-size 128 --hidden-dim 256 --fc-dropout 0.2 --rnn-dropout 0.2 --emb-dropout 0.05 --embedding-dim 300 --dataset-folder dataset/tig --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file scratch --emb-type "CNN" --trial-id $r --train-model --data-size .75 --step-gamma .94


# python ./train-class.py --accelerator "gpu" --exp-name tig-sent-scratch-100 --max_epochs 60 --batch-size 128 --hidden-dim 256 --fc-dropout 0.2 --rnn-dropout 0.2 --emb-dropout 0.05 --embedding-dim 300 --dataset-folder dataset/tig --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file scratch --emb-type "CNN" --trial-id $r --train-model --data-size 1.0 --step-gamma .94


# python ./train-class.py --accelerator "gpu" --exp-name tig-sent-scratch-10 --max_epochs 60 --batch-size 128 --hidden-dim 256 --fc-dropout 0.2 --rnn-dropout 0.2 --emb-dropout 0.05 --embedding-dim 300 --dataset-folder dataset/tig --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file scratch --emb-type "CNN" --trial-id $r --train-model --data-size 0.1 --step-gamma .94


# done


python ./train-class.py --accelerator "gpu" --exp-name tig-sent-scratch-10 --max_epochs 60 --batch-size 128 --hidden-dim 256 --fc-dropout 0.2 --rnn-dropout 0.2 --emb-dropout 0.05 --embedding-dim 300 --dataset-folder dataset/tig --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file scratch --emb-type "CNN" --data-size .1 --step-gamma .94 --no-train-model  --test-trial-ids 1-2-3-4-5


python ./train-class.py --accelerator "gpu" --exp-name tig-sent-scratch-25 --max_epochs 60 --batch-size 128 --hidden-dim 256 --fc-dropout 0.2 --rnn-dropout 0.2 --emb-dropout 0.05 --embedding-dim 300 --dataset-folder dataset/tig --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file scratch --emb-type "CNN" --data-size .25 --step-gamma .94 --no-train-model  --test-trial-ids 1-2-3-4-5


python ./train-class.py --accelerator "gpu" --exp-name tig-sent-scratch-50 --max_epochs 60 --batch-size 128 --hidden-dim 256 --fc-dropout 0.2 --rnn-dropout 0.2 --emb-dropout 0.05 --embedding-dim 300 --dataset-folder dataset/tig --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file scratch --emb-type "CNN" --data-size .50 --step-gamma .94 --no-train-model  --test-trial-ids 1-2-3-4-5


python ./train-class.py --accelerator "gpu" --exp-name tig-sent-scratch-75 --max_epochs 60 --batch-size 128 --hidden-dim 256 --fc-dropout 0.2 --rnn-dropout 0.2 --emb-dropout 0.05 --embedding-dim 300 --dataset-folder dataset/tig --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file scratch --emb-type "CNN" --data-size .75 --step-gamma .94 --no-train-model  --test-trial-ids 1-2-3-4-5


python ./train-class.py --accelerator "gpu" --exp-name tig-sent-scratch-100 --max_epochs 60 --batch-size 128 --hidden-dim 256 --fc-dropout 0.2 --rnn-dropout 0.2 --emb-dropout 0.05 --embedding-dim 300 --dataset-folder dataset/tig --charset-path "data/am-charset.txt" --max-seq-len 100 --num-rnn-layers 1 --learning-rate 0.001 --vector-file scratch --emb-type "CNN" --data-size .100 --step-gamma .94 --no-train-model  --test-trial-ids 1-2-3-4-5
