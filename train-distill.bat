@REM python train-distill.py --accelerator "gpu" --max_epochs 64 --fasttext-path "dataset/corpus/tig/tig-ft-w1.vec" --word2vec-path "dataset/corpus/tig/tig-w2v.vec" --corpus "dataset/corpus/tig/clean-tig-corpus.txt" --charset-path "data/am-charset.txt" --vector-load-ratio 1.0 --train-ratio 1.0 --exp-name "tig-distill-full32-both" --early-stop 0 --step_gamma 0.94 --neg_seq_len 32

set lang=am


python train-distill.py --accelerator "gpu" --max_epochs 64 --fasttext-path "dataset/corpus/%lang%/%lang%-ft.vec" --word2vec-path "dataset/corpus/%lang%/%lang%-w2v.vec" --corpus "dataset/corpus/%lang%/clean-%lang%-corpus.txt" --charset-path "data/am-charset.txt" --vector-load-ratio 0.5 --train-ratio 1.0 --exp-name "tig-distill-%lang%" --early-stop 0 --step_gamma 0.94 --neg_seq_len 32
@REM python train-distill.py --accelerator "gpu" --max_epochs 64 --fasttext-path "dataset/corpus/tig/tig-ft.vec" --word2vec-path "dataset/corpus/tig/tig-w2v.vec" --corpus "dataset/corpus/tig/clean-tig-corpus.txt" --charset-path "data/am-charset.txt" --vector-load-ratio 1.0 --train-ratio 0.9 --exp-name "tig-distill-w2v-early-ft" --early-stop 1