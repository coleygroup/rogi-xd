ALL = cv-orig cv-xd rogi

.PHONY: $(ALL) clean
.SILENT: $(ALL) clean cv-rand

all : $(ALL)

cv-orig:
	./scripts/cv.sh -v
	./scripts/cv.sh -f "random" -l 256 -v
	./scripts/cv.sh -f "random" -l 512 -v

cv-xd:
	./scripts/cv.sh
	./scripts/cv.sh -f "random" -l 256
	./scripts/cv.sh -f "random" -l 512

rogi:
	./scripts/rogi.sh

knn:
	./scripts/knn.sh
	
clean:
	rm -rf logs/*