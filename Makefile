#!make

SHELL:=bash

.PHONY: deb_requirements
deb_requirements:
	sudo apt-get install python-tk

.PHONY: pip_requirements
pip_requirements:
	pip install -r requirements.txt

.PHONY: requirements
requirements: deb_requirements pip_requirements

aclImdb_v1.tar.gz:
	wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

.PHONY:
classify:
	classify.py
