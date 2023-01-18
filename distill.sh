

lang=am

python train-distill.py --accelerator "gpu" --max_epochs 128 --fasttext-path "dataset/corpus/$lang/$lang-ft.vec" --word2vec-path "dataset/corpus/$lang/$lang-w2v.vec" --corpus "dataset/corpus/$lang/clean-$lang-corpus.txt" --charset-path "data/am-charset.txt" --vector-load-ratio 1.0 --train-ratio 1.0 --exp-name "distill-$lang" --early-stop 0 --step_gamma 0.94 --neg_seq_len 32 --model-size 'large'
