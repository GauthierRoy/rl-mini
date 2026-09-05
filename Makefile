.PHONY: install install-colab lint

install:
	pip install -r requirements.txt

install-colab:
	pip install -r requirements-colab.txt
	pip uninstall -y torchao || true
	python -c "import torch; assert torch.cuda.is_available(), 'no GPU: switch Colab runtime to T4'"
	python -c "from trl import GRPOTrainer; print('env OK')"

lint:
	bash scripts/lint.sh
