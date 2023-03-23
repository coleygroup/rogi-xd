ALL = cv-v1 cv-v2 rogi

.PHONY: $(ALL)
.SILENT: $(ALL)

all : $(ALL)

cv-v1:
	./scripts/cv.sh -v
	/scripts/cv.sh -v -r -f "chemgpt chemberta"

cv-v2:
	./scripts/cv.sh
	./scripts/cv.sh -r -f "chemgpt chemberta"

rogi:
	./scripts/rogi.sh
