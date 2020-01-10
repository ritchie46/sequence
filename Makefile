SHELL = /bin/bash

test:
	@pytest sequence/test/*.py

test-side-effects:
	@pytest sequence/test/side_effects