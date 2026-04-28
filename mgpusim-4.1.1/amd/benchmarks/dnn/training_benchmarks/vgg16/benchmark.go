// Package vgg16 implements VGG16 network training.
package vgg16

import (
	"math"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/dataset/imagenet"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/gputensor"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/gputraining"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/layers"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/training"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/training/optimization"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/mccl"
	"github.com/sarchlab/mgpusim/v4/amd/driver"
)

// Benchmark defines the VGG16 network training benchmark.
type Benchmark struct {
	driver   *driver.Driver
	ctx      *driver.Context
	to       []*gputensor.GPUOperator
	gpus     []int
	contexts []*driver.Context

	networks []training.Network
	trainer  gputraining.DataParallelismMultiGPUTrainer

	BatchSize          int
	Epoch              int
	MaxBatchPerEpoch   int
	EnableTesting      bool
	EnableVerification bool
	Lite               bool

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
	b.ctx = driver.Init()

	return b
}

// SelectGPU selects the GPU to use.
func (b *Benchmark) SelectGPU(gpuIDs []int) {
	b.gpus = gpuIDs
}

func (b *Benchmark) init() {
	for _, gpu := range b.gpus {
		b.defineNetwork(gpu)
	}

	b.createTrainer()
	b.randomizeParams()
}

func (b *Benchmark) defineNetwork(gpuID int) {
	context := b.driver.InitWithExistingPID(b.ctx)
	b.driver.SelectGPU(context, gpuID)
	to := gputensor.NewGPUOperator(b.driver, context)
	inputSize := imagenet.InputImageSize()

	if b.EnableVerification {
		to.EnableVerification()
	}

	var network training.Network
	if b.Lite {
		if inputSize%4 != 0 {
			panic("vgg16 lite requires -img-size divisible by 4")
		}

		s1 := inputSize / 2
		s2 := s1 / 2

		network = training.Network{
			Layers: []layers.Layer{
				layers.NewConv2D(0, to, []int{3, inputSize, inputSize}, []int{16, 3, 3, 3}, []int{1, 1}, []int{1, 1}),
				layers.NewReluLayer(to),
				layers.NewMaxPoolingLayer(to, []int{2, 2}, []int{0, 0}, []int{2, 2}),

				layers.NewConv2D(2, to, []int{16, s1, s1}, []int{32, 16, 3, 3}, []int{1, 1}, []int{1, 1}),
				layers.NewReluLayer(to),
				layers.NewMaxPoolingLayer(to, []int{2, 2}, []int{0, 0}, []int{2, 2}),

				layers.NewAvgPoolingLayer(to, []int{s2, s2}, []int{0, 0}, []int{s2, s2}),
				layers.NewFullyConnectedLayer(4, to, 32, 200),
			},
		}
	} else {
		if inputSize%32 != 0 {
			panic("vgg16 full requires -img-size divisible by 32")
		}

		s1 := inputSize / 2
		s2 := s1 / 2
		s3 := s2 / 2
		s4 := s3 / 2
		s5 := s4 / 2

		network = training.Network{
			Layers: []layers.Layer{
				layers.NewConv2D(0, to, []int{3, inputSize, inputSize}, []int{64, 3, 3, 3}, []int{1, 1}, []int{1, 1}),
				layers.NewReluLayer(to),
				layers.NewConv2D(2, to, []int{64, inputSize, inputSize}, []int{64, 64, 3, 3}, []int{1, 1}, []int{1, 1}),
				layers.NewReluLayer(to),
				layers.NewMaxPoolingLayer(to, []int{2, 2}, []int{0, 0}, []int{2, 2}),

				layers.NewConv2D(5, to, []int{64, s1, s1}, []int{128, 64, 3, 3}, []int{1, 1}, []int{1, 1}),
				layers.NewReluLayer(to),
				layers.NewConv2D(7, to, []int{128, s1, s1}, []int{128, 128, 3, 3}, []int{1, 1}, []int{1, 1}),
				layers.NewReluLayer(to),
				layers.NewConv2D(9, to, []int{128, s1, s1}, []int{128, 128, 3, 3}, []int{1, 1}, []int{1, 1}),
				layers.NewReluLayer(to),
				layers.NewMaxPoolingLayer(to, []int{2, 2}, []int{0, 0}, []int{2, 2}),

				layers.NewConv2D(12, to, []int{128, s2, s2}, []int{256, 128, 3, 3}, []int{1, 1}, []int{1, 1}),
				layers.NewReluLayer(to),
				layers.NewConv2D(14, to, []int{256, s2, s2}, []int{256, 256, 3, 3}, []int{1, 1}, []int{1, 1}),
				layers.NewReluLayer(to),
				layers.NewConv2D(16, to, []int{256, s2, s2}, []int{256, 256, 3, 3}, []int{1, 1}, []int{1, 1}),
				layers.NewReluLayer(to),
				layers.NewMaxPoolingLayer(to, []int{2, 2}, []int{0, 0}, []int{2, 2}),

				layers.NewConv2D(19, to, []int{256, s3, s3}, []int{512, 256, 3, 3}, []int{1, 1}, []int{1, 1}),
				layers.NewReluLayer(to),
				layers.NewConv2D(21, to, []int{512, s3, s3}, []int{512, 512, 3, 3}, []int{1, 1}, []int{1, 1}),
				layers.NewReluLayer(to),
				layers.NewConv2D(23, to, []int{512, s3, s3}, []int{512, 512, 3, 3}, []int{1, 1}, []int{1, 1}),
				layers.NewReluLayer(to),
				layers.NewMaxPoolingLayer(to, []int{2, 2}, []int{0, 0}, []int{2, 2}),

				layers.NewConv2D(26, to, []int{512, s4, s4}, []int{512, 512, 3, 3}, []int{1, 1}, []int{1, 1}),
				layers.NewReluLayer(to),
				layers.NewConv2D(27, to, []int{512, s4, s4}, []int{512, 512, 3, 3}, []int{1, 1}, []int{1, 1}),
				layers.NewReluLayer(to),
				layers.NewConv2D(29, to, []int{512, s4, s4}, []int{512, 512, 3, 3}, []int{1, 1}, []int{1, 1}),
				layers.NewReluLayer(to),
				layers.NewMaxPoolingLayer(to, []int{2, 2}, []int{0, 0}, []int{2, 2}),

				layers.NewFullyConnectedLayer(32, to, s5*s5*512, 2*2*512),
				layers.NewReluLayer(to),
				layers.NewFullyConnectedLayer(34, to, 2*2*512, 200),
			},
		}
	}

	b.networks = append(b.networks, network)
	b.contexts = append(b.contexts, context)
	b.to = append(b.to, to)
}

func (b *Benchmark) createTrainer() {
	sources := make([]training.DataSource, len(b.networks))
	alg := make([]optimization.Alg, len(b.networks))
	testers := make([]*training.Tester, len(b.networks))
	lossFuncs := make([]training.LossFunction, len(b.networks))

	for i := 0; i < len(b.networks); i++ {
		sources[i] = imagenet.NewTrainingDataSource(b.to[i])
		alg[i] = optimization.NewAdam(b.to[i], 0.001)
		lossFuncs[i] = training.NewSoftmaxCrossEntropy(b.to[i])

		if b.EnableTesting {
			testers[i] = &training.Tester{
				DataSource: imagenet.NewTestDataSource(b.to[i]),
				Network:    b.networks[i],
				BatchSize:  math.MaxInt32,
			}
		}
	}

	b.trainer = gputraining.DataParallelismMultiGPUTrainer{
		TensorOperators:  b.to,
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
		Contexts:         b.contexts,
		Driver:           b.driver,

		EnableEpochPageAllocTrace:    b.EnablePageAllocationTrace,
		PageAllocTraceDir:            b.PageAllocationTraceDir,
		EnableAutoPageReleaseDryRun:  b.EnableAutoPageReleaseDryRun,
		AutoPageReleaseDryRunDir:     b.AutoPageReleaseDryRunDir,
		EnableAutoPageReleaseEnforce: b.EnableAutoPageReleaseEnforce,
		AutoPageReleaseEnforceDir:    b.AutoPageReleaseEnforceDir,
	}
}

func (b *Benchmark) randomizeParams() {
	initNet := b.networks[0]
	for _, l := range initNet.Layers {
		l.Randomize()
	}

	gpuNum := len(b.networks)

	for i := range b.networks[0].Layers {
		if b.networks[0].Layers[i].Parameters() == nil {
			continue
		}

		params := make([]*gputensor.Tensor, gpuNum)
		datas := make([]driver.Ptr, gpuNum)

		for j := 0; j < gpuNum; j++ {
			params[j] = b.networks[j].Layers[i].Parameters().(*gputensor.Tensor)
		}

		dataSizeArr := params[0].Size()
		dataSize := 1
		for i := 0; i < len(dataSizeArr); i++ {
			dataSize *= dataSizeArr[i]
		}

		for i := 0; i < len(params); i++ {
			datas[i] = params[i].Ptr()
		}
		comms := mccl.CommInitAllMultipleContexts(
			gpuNum, b.driver, b.contexts, b.gpus)
		mccl.BroadcastRing(b.driver, comms, 1, datas, dataSize)
	}
}

// Run executes the benchmark.
func (b *Benchmark) Run() {
	b.init()
	b.trainer.Train()
}

// Verify runs the benchmark on the CPU and checks the result.
func (b *Benchmark) Verify() {
	panic("not implemented")
}

// SetUnifiedMemory asks the benchmark to use unified memory.
func (b *Benchmark) SetUnifiedMemory() {
	panic("unified memory is not supported by dnn workloads")
}
