SHELL = /bin/bash

test:
	@pytest sequence/test/*.py

test-train:
	@pytest sequence/test/train