# adaboost
adaboost algorithms for multilabel, multiclass problem. Details in "Improved Boosting Algorithms Using Confidence-rated Predictions"

## Algorithms
- Adaboost.MH:Using Hamming Loss for Multilabel, multiclass Problems.
- Adaboost.MR:Using Ranking Loss for Multilabel, multiclass Problems.

## Run the code
```
python -m adaboost.models.adaboost_MH --data_dir= --epoch_num= --results_dir= [--VIS]
```
- data_dir: the absolute path of the directory that saves your datasets.
- epoch_num: the number of iterations.
- results_dir: save the predicted results of the test set.
- VIS: if VIS is set, the prediction accuracy of all iterations are plotted by ggplot. 

## Other things
We use [yeast](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html) to test the two algorithms. If you use a different dataset, you may add a function for data parsing in `data` and modify `data.\_\_init\_\_` accordingly.
