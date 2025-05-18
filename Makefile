clean-logs-all:
	rm -rf logs

clean-logs-latest:
	@latest_dir=$$(ls -1d logs/* | sort | tail -n 1) && \
	echo "Removing latest directory: $$latest_dir" && \
	rm -rf "$$latest_dir"

test-train-infer:
	python3 . model train -p 32 -b 100 --testing -l 0.01 -e 10000 && \
	python3 . model infer -p 32 -b 100 --testing -c
