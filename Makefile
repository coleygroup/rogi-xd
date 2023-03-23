.PHONY: v1 v2

v1:
	./scripts/cv-getopts.sh -v

v2:
	./scripts/cv-getopts.sh

v1-reinit:
	./scripts/cv-getopts.sh -v -r -f "chemgpt chemberta"

v2-reinit:
	./scripts/cv-getopts.sh -r -f "chemgpt chemberta"