# Nsight Systems flamegraph adapter

Script for parsing Nsight Systems report files containing CPU call stacks and producing an output suitable for `flamegraph.pl`.

## Install FlameGraph

To visualize  with FlameGraph call stack download `flamegraph.pl` from the [flamegraph repository](https://github.com/brendangregg/FlameGraph).

### On Windows

To use `flamegraph.pl` on Windows, you will need to [install Perl](https://www.perl.org/get.html).

## Function names

By default, script tries to shorten function definitions (eliminating return type, arguments and templates). In some complex cases shortening may fail and return a full function definition. To disable shortening defining `--full_function_names=False` argument can be used.

## Usage

Generating flamegraph from Nsight Systems report file:

### Unix

```bash
python3 stackcollapse_nsys.py report.nsys-rep | ./flamegraph.pl > result_flamegraph.svg
```

### Windows

```bash
PowerShell -Command "python stackcollapse_nsys.py report.nsys-rep | perl flamegraph.pl > result_flamegraph.svg"
```

### Parameters

The following parameters can be passed to the script:

| Parameter             | Short | Description                                                                                                        | Default value                                           |
| --------------------- | ----- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------- |
| --nsys                |       | Path to the NVIDIA Nsight Systems CLI directory (for ex.: `/opt/nvidia/nsight-systems/2022.4.1/target-linux-x64`). | Current NVIDIA Nsight Systems CLI installation location |
| --out                 | -o    | Path to a result file containing a data suitable for `flamegraph.pl`.                                              | Output is written to stdout                             |
| --full_function_names |       | Use full function names with return type, arguments and expanded templates, if available.                          | False                                                   |
