#!make

SHELL:=bash

.PHONY: deb_requirements
deb_requirements:
	sudo apt-get install python-tk pep8 pylint

.PHONY: pip_requirements
pip_requirements:
	pip install -r requirements.txt

.PHONY: requirements
requirements: deb_requirements pip_requirements

aclImdb_v1.tar.gz:
	wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

.PHONY:
classify:
	./run.py

.PHONY: lint
lint:
	pycodestyle movie_sentiment
	pylint --rcfile=.pylintrc movie_sentiment

.PHONY: dos2unix
dos2unix:
	find . -name "*.py" -exec dos2unix {} \;
