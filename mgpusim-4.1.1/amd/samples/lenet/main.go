package main

import (
	"flag"
	"math/rand"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/training_benchmarks/lenet"

	"github.com/sarchlab/mgpusim/v4/amd/samples/runner"
)

var epochFlag = flag.Int("epoch", 1, "Number of epoch to run.")
var maxBatchPerEpochFlag = flag.Int("max-batch-per-epoch", 2,
	"Number of epochs to run.")
var batchSizeFlag = flag.Int("batch-size", 32,
	"Number of images per batch")
var enableTestingFlag = flag.Bool("enable-testing", false,
	"If enable testing is set, the trainer will evaluate the trained model after each epoch")
var enableVerification = flag.Bool("enable-verification", false,
	`If set, all tenser operations will be verified against CPU results. Do not 
turn on if you care about the final results. This flag will introduce extra
GPU-to-CPU memory copies.`)
var enablePageAllocationTraceFlag = flag.Bool("enable-page-allocation-trace", true,
	"Enable per-epoch page allocation trace CSV output.")
var pageAllocationTraceDirFlag = flag.String("page-allocation-trace-dir",
	"lenet_page_alloc_trace_off",
	"Directory for page allocation trace CSV files.")
var enableAutoPageReleaseDryRunFlag = flag.Bool("enable-auto-page-release-dry-run", true,
	"Enable auto page release dry-run mode.")
var autoPageReleaseDryRunDirFlag = flag.String("auto-page-release-dry-run-dir",
	"lenet_auto_release_dry_run",
	"Directory for auto page release dry-run CSV files.")
var enableAutoPageReleaseEnforceFlag = flag.Bool("enable-auto-page-release-enforce", false,
	"Enable auto page release enforce mode.")
var autoPageReleaseEnforceDirFlag = flag.String("auto-page-release-enforce-dir",
	"lenet_auto_release_enforce_off",
	"Directory for auto page release enforce CSV files.")

func main() {
	rand.Seed(1)
	flag.Parse()

	runner := new(runner.Runner).Init()

	benchmark := lenet.NewBenchmark(runner.Driver())
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

	runner.AddBenchmark(benchmark)

	runner.Run()
}
