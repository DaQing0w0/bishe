package main

import (
	"flag"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/training_benchmarks/vgg16"
	"github.com/sarchlab/mgpusim/v4/amd/samples/runner"
)

var epochFlag = flag.Int("epoch", 1, "Number of epoch to run.")
var maxBatchPerEpochFlag = flag.Int("max-batch-per-epoch", 2,
	"Number of epochs to run.")
var batchSizeFlag = flag.Int("batch-size", 8,
	"Number of images per batch")
var enableTestingFlag = flag.Bool("enable-testing", false,
	"If set, the trainer will evaluate the trained model after each epoch")
var enableVerification = flag.Bool("enable-verification", false,
	`If set, all tenser operations will be verified against CPU results. Do not 
turn on if you care about the final results. This flag will introduce extra
GPU-to-CPU memory copies.`)
var enablePageAllocationTraceFlag = flag.Bool("enable-page-allocation-trace", true,
	"Enable per-epoch page allocation trace CSV output.")
var pageAllocationTraceDirFlag = flag.String("page-allocation-trace-dir",
	"vgg16_page_alloc_trace_off",
	"Directory for page allocation trace CSV files.")
var enableAutoPageReleaseDryRunFlag = flag.Bool("enable-auto-page-release-dry-run", false,
	"Enable auto page release dry-run mode.")
var autoPageReleaseDryRunDirFlag = flag.String("auto-page-release-dry-run-dir",
	"vgg16_auto_release_dry_run",
	"Directory for auto page release dry-run CSV files.")
var enableAutoPageReleaseEnforceFlag = flag.Bool("enable-auto-page-release-enforce", true,
	"Enable auto page release enforce mode.")
var autoPageReleaseEnforceDirFlag = flag.String("auto-page-release-enforce-dir",
	"vgg16_auto_release_enforce_off",
	"Directory for auto page release enforce CSV files.")
var liteFlag = flag.Bool("lite", false,
	"Use a smaller VGG-style network with fewer layers for faster runs.")

func main() {
	flag.Parse()

	runner := new(runner.Runner).Init()

	benchmark := vgg16.NewBenchmark(runner.Driver())
	benchmark.Epoch = *epochFlag
	benchmark.MaxBatchPerEpoch = *maxBatchPerEpochFlag
	benchmark.BatchSize = *batchSizeFlag
	benchmark.EnableTesting = *enableTestingFlag
	benchmark.EnableVerification = *enableVerification
	benchmark.EnablePageAllocationTrace = *enablePageAllocationTraceFlag
	benchmark.PageAllocationTraceDir = *pageAllocationTraceDirFlag
	benchmark.EnableAutoPageReleaseDryRun = *enableAutoPageReleaseDryRunFlag
	benchmark.AutoPageReleaseDryRunDir = *autoPageReleaseDryRunDirFlag
	benchmark.EnableAutoPageReleaseEnforce = *enableAutoPageReleaseEnforceFlag
	benchmark.AutoPageReleaseEnforceDir = *autoPageReleaseEnforceDirFlag
	benchmark.Lite = *liteFlag

	runner.AddBenchmark(benchmark)

	runner.Run()
}
