# KVQuant Deployment Code

The code in this folder can be used to run the inference experiments from the paper (for benchmarking kernel runtime).

## Installation

1. Create a conda environment
```
conda create --name benchmark python=3.9 -y
conda activate benchmark
```

2. Clone and install the dependencies (including the local transformers environment)
```
pip install transformers
pip install torch datasets sentencepiece scikit-learn protobuf
pip install accelerate -U
pip install -e .
cd kvquant
python setup_cuda.py install
cd ..
pip install flash-attn --no-build-isolation
```

3. Run kernel benchmarking

Note that the quantizer is obtained from steps in the quant directory.

```
cp ../quant/quantizers.pickle .
CUDA_VISIBLE_DEVICES=0 python cache-llama-activations.py <path-to-llama-7b-hf> --wbits 4 --nsamples 1 --seqlen 2048 --quantizer-path quantizers.pickle --output-path activations-seqlen2048.pickle;
```

Assuming the activations and the quantizers are stored in "activations.pickle" and "quantizers.pickle", you can run the kernel benchmarks by running the python scripts in the "scripts" folder.


## Batch benchmark runner

`run_all.py` sequentially runs the current CUDA kernel experiments, captures each raw profiler table to a log, and converts profiler/stdout metrics to a long-form CSV. It also records runner wall time and whether the experiment passed, failed, or was skipped because prerequisite artifacts are missing.

Before running the KVQuant key/value experiments, ensure these files exist in the `benchmarking/` directory:

```bash
quantizers.pickle
activations-seqlen2048.pickle
```

List the available experiments:

```bash
python run_all.py --list
```

Dry-run all experiments and show prerequisite checks:

```bash
python run_all.py --dry-run
```

If you want the generated dashboard, install the optional plotting dependency first:

```bash
pip install -e ".[plotting]"
```

Run everything, write `benchmark_results/<timestamp>/results.csv`, summarize metrics into `summary.csv`, and draw `benchmark_summary.png`:

```bash
python run_all.py --plot
```

Useful options include `--only NAME`, `--exclude NAME`, `--repeat N`, `--timeout SECONDS`, `--stop-on-failure`, and `--append`. To summarize an existing CSV later:

```bash
python summarize_benchmarks.py benchmark_results/<timestamp>/results.csv
```

The dashboard reports mean runner wall time and the largest `cuda_time_avg` entry per run. Average per-call CUDA time is used because the experiments issue different numbers of kernel invocations; summing profiler rows would double-count parent/child entries and would not be directly comparable. The complete per-kernel data remains available in `results.csv`.

### Generate benchmark artifacts through ModelScope

If HuggingFace is unreachable, the wikitext2 loader can use ModelScope. Install ModelScope in each environment that loads calibration data:

```bash
pip install modelscope
```

Then pass `--dataset_source modelscope` to `run-fisher.py`, `llama_simquant.py`, or `cache-llama-activations.py`. By default `run_all.py --prepare-artifacts` uses ModelScope. The ModelScope dataset ID defaults to `modelscope/wikitext`; override it with:

```bash
export KVQUANT_MODELSCOPE_WIKITEXT2_DATASET=<valid-modelscope-wikitext2-dataset-id>
```

The integrated artifact generation command is:

```bash
python run_all.py \
  --prepare-artifacts \
  --model-path /path/to/llama-7b-hf \
  --gradient-python /path/to/grad-env/bin/python \
  --quant-python /path/to/kvquant-env/bin/python \
  --plot
```

It sequentially generates Fisher information, `quantizers.pickle`, and `activations-seqlen2048.pickle`, then runs the benchmark experiments. Existing artifacts are reused unless `--force-artifacts` is passed.
