// Package xor implements a extremely simple network that can perform the xor
// operation.
package xor

import (
	"fmt"
	"math"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/gputensor"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/gputraining"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/layers"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/training"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/training/optimization"
	"github.com/sarchlab/mgpusim/v4/amd/driver"
)

// Benchmark defines the XOR network training benchmark.
type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	to      *gputensor.GPUOperator

	gpus []int

	networks []training.Network
	trainer  gputraining.DataParallelismMultiGPUTrainer

	BatchSize          int
	Epoch              int
	MaxBatchPerEpoch   int
	EnableTesting      bool
	EnableVerification bool

	EnablePageAllocationTrace    bool
	PageAllocationTraceDir       string
	EnableAutoPageReleaseDryRun  bool
	AutoPageReleaseDryRunDir     string
	EnableAutoPageReleaseEnforce bool
	AutoPageReleaseEnforceDir    string
}

// NewBenchmark creates a new benchmark.
func NewBenchmark(driver *driver.Driver) *Benchmark {
	b := new(Benchmark)

	b.driver = driver
	b.context = b.driver.Init()
	b.to = gputensor.NewGPUOperator(b.driver, b.context)
	b.EnableVerification = true

	b.Epoch = 50
	b.BatchSize = 4
	b.MaxBatchPerEpoch = math.MaxInt32

	b.networks = []training.Network{
		{
			Layers: []layers.Layer{
				layers.NewFullyConnectedLayer(
					0,
					b.to,
					2, 4,
				),
				layers.NewReluLayer(b.to),
				layers.NewFullyConnectedLayer(
					2,
					b.to,
					4, 2,
				),
			},
		},
	}

	b.enableLayerVerification(&b.networks[0])

	return b
}

func (b *Benchmark) enableLayerVerification(network *training.Network) {

}

// SelectGPU selects the GPU to use.
func (b *Benchmark) SelectGPU(gpuIDs []int) {
	if len(gpuIDs) > 1 {
		panic("multi-GPU is not supported by DNN workloads")
	}
	b.gpus = gpuIDs
}

func (b *Benchmark) createTrainer() {
	sources := []training.DataSource{NewDataSource(b.to)}
	alg := []optimization.Alg{optimization.NewAdam(b.to, 0.03)}
	lossFuncs := []training.LossFunction{training.NewSoftmaxCrossEntropy(b.to)}
	testers := make([]*training.Tester, 1)

	if b.EnableTesting {
		testers[0] = &training.Tester{
			DataSource: NewDataSource(b.to),
			Network:    b.networks[0],
			BatchSize:  math.MaxInt32,
		}
	}

	b.trainer = gputraining.DataParallelismMultiGPUTrainer{
		TensorOperators:  []*gputensor.GPUOperator{b.to},
		DataSource:       sources,
		Networks:         b.networks,
		LossFunc:         lossFuncs,
		OptimizationAlg:  alg,
		Tester:           testers,
		Epoch:            b.Epoch,
		MaxBatchPerEpoch: b.MaxBatchPerEpoch,
		BatchSize:        b.BatchSize,
		ShowBatchInfo:    true,
		GPUs:             b.gpus,
		Contexts:         []*driver.Context{b.context},
		Driver:           b.driver,

		EnableEpochPageAllocTrace:    b.EnablePageAllocationTrace,
		PageAllocTraceDir:            b.PageAllocationTraceDir,
		EnableAutoPageReleaseDryRun:  b.EnableAutoPageReleaseDryRun,
		AutoPageReleaseDryRunDir:     b.AutoPageReleaseDryRunDir,
		EnableAutoPageReleaseEnforce: b.EnableAutoPageReleaseEnforce,
		AutoPageReleaseEnforceDir:    b.AutoPageReleaseEnforceDir,
	}
}

// Run executes the benchmark.
func (b *Benchmark) Run() {
	if b.EnableVerification {
		b.to.EnableVerification()
	}

	if len(b.gpus) == 1 {
		b.driver.SelectGPU(b.context, b.gpus[0])
	}

	b.createTrainer()

	for _, l := range b.networks[0].Layers {
		l.Randomize()
	}

	b.trainer.Train()
}

func (b *Benchmark) printLayerParams() {
	for i, l := range b.networks[0].Layers {
		params := l.Parameters()
		if params != nil {
			fmt.Println("Layer ", i, params.Vector())
		}
	}
}

// Verify runs the benchmark on the CPU and checks the result.
func (b *Benchmark) Verify() {
	panic("not implemented")
}

// SetUnifiedMemory asks the benchmark to use unified memory.
func (b *Benchmark) SetUnifiedMemory() {
	panic("unified memory is not supported by dnn workloads")
}
