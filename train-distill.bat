python train-distill.py --accelerator "gpu" --max_epochs 60 --fasttext-path "dataset/corpus/tig/tig-ft.vec" --word2vec-path "dataset/corpus/tig/tig-w2v.vec" --corpus "dataset/corpus/tig/clean-tig-corpus.txt" --charset-path "data/am-charset.txt" --vector-load-ratio 1.0 --train-ratio 0.9 --exp-name "tig-distill" --early-stop 0

python train-distill.py --accelerator "gpu" --max_epochs 60 --fasttext-path "dataset/corpus/tig/tig-ft.vec" --word2vec-path "dataset/corpus/tig/tig-w2v.vec" --corpus "dataset/corpus/tig/clean-tig-corpus.txt" --charset-path "data/am-charset.txt" --vector-load-ratio 1.0 --train-ratio 0.9 --exp-name "tig-distill" --early-stop 1