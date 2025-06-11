no-op:
	echo no-op

device = cuda
tensor = torch
points = 32
offset = 0
batch = 100
alpha = 0.01
epochs = 10000
args =

install-requirements:
	pip install -r requirements.txt

test-train-infer:
	python3 . model train -t $(tensor) -d $(device) -p $(points) -o $(offset) -b $(batch) --testing -l $(alpha) -e $(epochs) && \
	python3 . model infer -t $(tensor) -d $(device) -p $(points) -o $(offset) -b $(batch) --testing -c

train:
	python3 . model train -t $(tensor) -d $(device) -p $(points) -o $(offset) -b $(batch) -l $(alpha) -e $(epochs) $(args)

infer:
	python3 . model infer -t $(tensor) -d $(device) -p $(points) -o $(offset) -b $(batch) -c

clean-logs-all:
	rm -rf _generated/logs

clean-logs-latest:
	@latest_dir=$$(ls -1d _generated/logs/* | sort | tail -n 1) && \
	echo "Removing latest directory: $$latest_dir" && \
	rm -rf "$$latest_dir"

# Build a PNG ready for readme inclusion complete with border.
# Should be viewable in both dark and bright mode in github.
# Default dpi of 120 seems to display well on github.
name = log-reg
dpi = 120
tex-png:
	scripts/make_tex_png.sh $(name) $(dpi)
