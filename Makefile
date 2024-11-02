clean: clean-pyc clean-test
quality: set-style-dep check-quality
style: set-style-dep set-style
setup: set-style-dep set-test-dep set-git set-dev set-output
test: set-test-dep set-test
dataset: get-dataset


##### basic #####
set-git:
	git config --local commit.template .gitmessage

set-style-dep:
	pip3 install isort==5.12.0 black==23.3.0 flake8==4.0.1

set-test-dep:
	pip3 install pytest==7.0.1

set-dev:
	pip3 install -r requirements.txt

set-test:
	python3 -m pytest tests/

set-style:
	black --config pyproject.toml .
	isort --settings-path pyproject.toml .
	flake8 .

check-quality:
	black --config pyproject.toml --check .
	isort --settings-path pyproject.toml --check-only .
	flake8 .

#####  clean  #####
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*
	rm -rf .pytest_cache
	rm -rf .mypy_cache

get-dataset:
	pip3 install gdown
	gdown https://drive.google.com/uc?id=16zAeGDmqbAvn7Iy8V-mBylKx6rG-wgLD
	gdown https://drive.google.com/uc?id=1cqxSVFxfonx5qKVIdfByOq7uYdNZY8ea
	gdown https://drive.google.com/uc?id=1zmfrXzT9lnLg7NlQ-hXekZlyX9aGNNqj
	mkdir -p 'dataset/train/clean'
	mkdir -p 'dataset/train/scan'
	mkdir -p 'dataset/test/scan'
	unzip 'train_clean.zip' -d 'dataset/train/clean'
	unzip 'train_scan.zip' -d 'dataset/train/scan'
	unzip 'test_scan.zip' -d 'dataset/test/scan'
	rm 'train_clean.zip'
	rm 'train_scan.zip'
	rm 'test_scan.zip'

set-output:
	mkdir -p output
	mkdir -p logs
	mkdir -p model_output
	mkdir -p submission
