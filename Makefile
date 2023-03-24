ALL = cv-v1 cv-v2 rogi

.PHONY: $(ALL) clean
.SILENT: $(ALL) clean cv-rand

all : $(ALL)

cv-v1:
	./scripts/cv.sh -v
	./scripts/cv.sh -f "random" -l 256 -v
	./scripts/cv.sh -f "random" -l 512 -v

cv-v2:
	./scripts/cv.sh
	./scripts/cv.sh -f "random" -l 256
	./scripts/cv.sh -f "random" -l 512

rogi:
	./scripts/rogi.sh

clean:
	rm -rf logs/*