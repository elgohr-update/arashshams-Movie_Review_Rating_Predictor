# Group 26: Movie Review Rating Predictor Pipeline
# author: Yuanzhe Marco Ma, Arash Shamseddini, Kaicheng Tan, Zhenrui Yu
# date: 2020-12-02

all : doc/imdb_rating_EDA.html doc/imdb_rating_predict_report.html results/dependency_diagram_makefile.png

# Download raw data
data/raw/Dataset_IMDB.csv: src/download_data.py
	# Download raw data from the original repository
	python src/download_data.py https://github.com/nproellochs/SentimentDictionaries/raw/master/Dataset_IMDB.csv data/raw/Dataset_IMDB.csv

# Pre-process data
data/processed/train.csv data/processed/test.csv: src/pre_process.py data/raw/Dataset_IMDB.csv
	# Pre-process data
	python src/pre_process.py --input_file=data/raw/Dataset_IMDB.csv --out_dir=data/processed

# Generate resources for EDA
results/histogram_rating_distribution.svg results/boxplot_rating_critics.svg results/histogram_rating_vs_text_length.svg doc/imdb_rating_EDA.html: src/eda_imdb.py doc/imdb_rating_EDA.ipynb data/processed/train.csv
	# Generate visualizations necessary for analysis reports
	python src/eda_imdb.py data/processed/train.csv results
	jupyter nbconvert --to notebook --inplace --execute doc/imdb_rating_EDA.ipynb
	jupyter nbconvert --to html doc/imdb_rating_EDA.ipynb

# Model tuning and fitting
results/model.pkl results/hyper_param_search_result.csv: src/imdb_rating_predict_model.py data/processed/train.csv
	# Generate and persist ML model
	python src/imdb_rating_predict_model.py data/processed/train.csv results

# Model evaluation
results/model_test_scores.csv results/true_vs_predict.svg: src/imdb_rating_test_results.py results/model.pkl data/processed/test.csv
	# Evaluate the model with test set
	python src/imdb_rating_test_results.py results/model.pkl data/processed/test.csv results

# Generate report
doc/imdb_rating_predict_report.html: results/histogram_rating_distribution.svg results/boxplot_rating_critics.svg results/histogram_rating_vs_text_length.svg
doc/imdb_rating_predict_report.html: results/model_test_scores.csv results/true_vs_predict.svg
doc/imdb_rating_predict_report.html:
	jupyter nbconvert --to notebook --inplace --execute doc/imdb_rating_predict_report.ipynb
	jupyter nbconvert --to html doc/imdb_rating_predict_report.ipynb

# Generate report for model comparison
doc/model_comparison.html:
	jupyter nbconvert --to html doc/model_comparison.ipynb

# Generate dependency diagram for Makefile
results/dependency_diagram_makefile.png: Makefile
	makefile2graph > Makefile.dot
	dot -Tpng Makefile.dot -o results/dependency_diagram_makefile.png
	rm -rf Makefile.dot

# Clean intermediate files
clean :
	rm -rf data/*
	rm -rf results/*
	rm -rf doc/*.html
