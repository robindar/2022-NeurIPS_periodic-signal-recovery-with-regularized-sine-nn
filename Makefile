default:

# Number of repetitions for the frequency estimation experiment
REP=015

all: images/single-tone.png images/multi-tone.png images/freq-estimation-noise-resistance_E11b.png images/freq-estimation-noise-resistance_E11c.png

clean:
	rm -rf src/__pycache__ src/structures_d/__pycache__

dataclean:
	rm -f data/*.yml data/*.done

distclean: clean dataclean
	rm -f images/*.png


data/E0%.done:
	python src/train.py $(shell basename -s .done $@) || exit 1
	@@touch $@

data/E11b.done data/E11c.done:
	for e in $$(echo {000..${REP}}); do for n in $$(echo {00..09}); do python src/train.py $(shell basename -s .done $@)_$${e}_$${n} || exit 1; done; done
	@@touch $@

images/fourier.png: data/E04_B3.done data/E04_C3.done data/E04_D3.done data/E04_E3.done
	python src/fourier-plots.py $(patsubst %.done,%.yml,$^)

images/freq-estimation-noise-resistance_E11b.png: data/E11b.done
	python src/frequency-estimation-noise-resistance.py 11b

images/freq-estimation-noise-resistance_E11c.png: data/E11c.done
	python src/frequency-estimation-noise-resistance.py 11c

images/single-tone.png: $(patsubst %, data/E05_%2.done, B C D E A Z Y X)
	python src/plot-periodic-recovery.py single-tone $(patsubst %.done,%.yml, $^)

images/multi-tone.png: $(patsubst %, data/E06_%2.done, B C E A Z)
	python src/plot-periodic-recovery.py multi-tone $(patsubst %.done,%.yml, $^)
