// Package runner defines how default benchmark samples are executed.
package runner

import (
	"log"
	"os"
	"strings"

	// Enable profiling
	_ "net/http/pprof"
	"sync"

	"github.com/sarchlab/akita/v4/mem/mem"
	"github.com/sarchlab/akita/v4/sim"
	"github.com/sarchlab/akita/v4/simulation"
	"github.com/sarchlab/akita/v4/tracing"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks"
	"github.com/sarchlab/mgpusim/v4/amd/driver"
	"github.com/sarchlab/mgpusim/v4/amd/samples/runner/emusystem"
	"github.com/sarchlab/mgpusim/v4/amd/samples/runner/timingconfig"
	"github.com/sarchlab/mgpusim/v4/amd/sampling"
)

type verificationPreEnablingBenchmark interface {
	benchmarks.Benchmark

	EnableVerification()
}

// Runner is a class that helps running the benchmarks in the official samples.
type Runner struct {
	simulation *simulation.Simulation
	platform   *sim.Domain
	reporter   *reporter

	Timing           bool
	Verify           bool
	Parallel         bool
	UseUnifiedMemory bool

	GPUIDs     []int
	benchmarks []benchmarks.Benchmark

	memLog    *os.File
	memTracer *addrSeqTracer
}

// epochAware is an interface for trainers that are aware of the current epoch.
type epochAware interface {
	CurrentEpoch() int
}

// 只记录地址序列的轻量 tracer
type addrSeqTracer struct {
	logger      *log.Logger
	timeTeller  sim.TimeTeller
	epochGetter func() int
}

// get address(inspired by akita/mem/trace/tracer.go)
func (t *addrSeqTracer) StartTask(task tracing.Task) {
	if strings.Contains(task.ID, "req_out") { // 跳过 req_out
		return
	}
	if req, ok := task.Detail.(mem.AccessReq); ok {
		ts := t.timeTeller.CurrentTime()
		epoch := -1
		if t.epochGetter != nil {
			epoch = t.epochGetter()
		}
		// CSV: time,task_id,address,bytes,epoch
		t.logger.Printf("%.12f,%s,0x%x,%d,%d\n",
			ts, task.ID, req.GetAddress(), req.GetByteSize(), epoch)
	}
}

// DO NOTHING
func (t *addrSeqTracer) StepTask(task tracing.Task)       {}
func (t *addrSeqTracer) AddMilestone(m tracing.Milestone) {}
func (t *addrSeqTracer) EndTask(task tracing.Task)        {}

// Init initializes the platform simulate
func (r *Runner) Init() *Runner {
	r.parseFlag()

	log.SetFlags(log.Llongfile | log.Ldate | log.Ltime)

	r.initSimulation()

	if r.Timing {
		r.buildTimingPlatform()
	} else {
		r.buildEmuPlatform()
	}

	r.createUnifiedGPUs()

	return r
}

func (r *Runner) initSimulation() {
	builder := simulation.MakeBuilder()

	if *parallelFlag {
		builder = builder.WithParallelEngine()
	}

	r.simulation = builder.Build()
}

func (r *Runner) buildEmuPlatform() {
	b := emusystem.MakeBuilder().
		WithSimulation(r.simulation).
		WithNumGPUs(r.GPUIDs[len(r.GPUIDs)-1])

	if *isaDebug {
		b = b.WithDebugISA()
	}

	r.platform = b.Build()
}

func (r *Runner) buildTimingPlatform() {
	sampling.InitSampledEngine()

	b := timingconfig.MakeBuilder().
		WithSimulation(r.simulation).
		WithNumGPUs(r.GPUIDs[len(r.GPUIDs)-1])

	if *magicMemoryCopy {
		b = b.WithMagicMemoryCopy()
	}

	r.platform = b.Build()
	r.reporter = newReporter(r.simulation)
	r.configureVisTracing()
	f, err := os.Create("mem.csv")
	if err != nil {
		panic(err)
	}
	r.memLog = f
	// 写 CSV 表头
	_, _ = f.WriteString("time,task_id,address,bytes,epoch\n")

	logger := log.New(f, "", 0)
	memTracer := &addrSeqTracer{
		logger:     logger,
		timeTeller: r.simulation.GetEngine(),
	}
	r.memTracer = memTracer

	// 按名称过滤并挂载组件
	filters := []string{"L1VAddrTrans"}
	hooked := 0
	for _, c := range r.simulation.Components() {
		okMatch := false
		for _, s := range filters {
			if strings.Contains(c.Name(), s) {
				okMatch = true
				break
			}
		}
		if !okMatch {
			continue
		}
		if hookable, ok := c.(tracing.NamedHookable); ok {
			tracing.CollectTrace(hookable, memTracer)
			hooked++
		}
	}
	log.Printf("[memtrace] hooked %d components", hooked)
}

func (r *Runner) configureVisTracing() {
	if !*visTracing {
		return
	}

	visTracer := r.simulation.GetVisTracer()
	for _, comp := range r.simulation.Components() {
		tracing.CollectTrace(comp.(tracing.NamedHookable), visTracer)
	}
}

func (r *Runner) createUnifiedGPUs() {
	if *unifiedGPUFlag == "" {
		return
	}

	driver := r.simulation.GetComponentByName("Driver").(*driver.Driver)
	unifiedGPUID := driver.CreateUnifiedGPU(nil, r.GPUIDs)
	r.GPUIDs = []int{unifiedGPUID}
}

// AddBenchmark adds an benchmark that the driver runs
func (r *Runner) AddBenchmark(b benchmarks.Benchmark) {
	b.SelectGPU(r.GPUIDs)
	if r.UseUnifiedMemory {
		b.SetUnifiedMemory()
	}

	// 绑定 epochGetter（如果benchmark支持）
	if r.memTracer != nil {
		if ea, ok := b.(epochAware); ok {
			r.memTracer.epochGetter = ea.CurrentEpoch
		}
	}

	r.benchmarks = append(r.benchmarks, b)
}

// AddBenchmarkWithoutSettingGPUsToUse allows for user specified GPUs for
// the benchmark to run.
func (r *Runner) AddBenchmarkWithoutSettingGPUsToUse(b benchmarks.Benchmark) {
	if r.UseUnifiedMemory {
		b.SetUnifiedMemory()
	}

	if r.memTracer != nil {
		if ea, ok := b.(epochAware); ok {
			r.memTracer.epochGetter = ea.CurrentEpoch
		}
	}

	r.benchmarks = append(r.benchmarks, b)
}

// Run runs the benchmark
func (r *Runner) Run() {
	r.Driver().Run()

	var wg sync.WaitGroup
	for _, b := range r.benchmarks {
		wg.Add(1)
		go func(b benchmarks.Benchmark, wg *sync.WaitGroup) {
			if r.Verify {
				if b, ok := b.(verificationPreEnablingBenchmark); ok {
					b.EnableVerification()
				}
			}

			b.Run()

			if r.Verify {
				b.Verify()
			}
			wg.Done()
		}(b, &wg)
	}
	wg.Wait()

	if r.reporter != nil {
		r.reporter.report()
	}

	r.Driver().Terminate()
	r.simulation.Terminate()

	if r.memLog != nil {
		r.memLog.Close()
	}
}

// Driver returns the GPU driver used by the current runner.
func (r *Runner) Driver() *driver.Driver {
	return r.simulation.GetComponentByName("Driver").(*driver.Driver)
}

// Engine returns the event-driven simulation engine used by the current runner.
func (r *Runner) Engine() sim.Engine {
	return r.simulation.GetEngine()
}
