run:
	./run.sh

run-knn:
	python kNN/knn/train_and_evaluate.py

.PHONY: clean
clean:
	find . -name '*.pyc' -delete
