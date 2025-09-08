python main.py train -t data/train.fa -v data/valid.fa -e 30 -s 201 -a RNAModificationModel -l 0.0002
python main.py eval -d data/test.fa -s 201 -m save_model/RNAModificationModel_best_model.pth -a RNAModificationModel
