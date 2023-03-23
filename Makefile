.PHONY: v1 v1-reinit v2 v2-reinit rogi

all : v1 v1-reinit v2 v2-reinit rogi

v1:
	./scripts/cv.sh -v

v2:
	./scripts/cv.sh

v1-reinit:
	./scripts/cv.sh -v -r -f "chemgpt chemberta"

v2-reinit:
	./scripts/cv.sh -r -f "chemgpt chemberta"

rogi:
	./scripts/rogi.sh
