package driver

import (
	"strconv"
	"testing"

	"github.com/sarchlab/akita/v4/mem/vm"
	"github.com/sarchlab/akita/v4/sim"
)

func newAccessTracerForBench() *epochPageAllocTracer {
	t := newEpochPageAllocTracer("", func(uint64) {})
	pid := vm.PID(1)
	epoch := 2
	pageVAddr := uint64(0x1000)

	t.activeEpochByPID[pid] = epoch
	if _, ok := t.seqByEpochPIDVAddr[epoch]; !ok {
		t.seqByEpochPIDVAddr[epoch] = make(map[vm.PID]map[uint64]uint64)
	}
	if _, ok := t.seqByEpochPIDVAddr[epoch][pid]; !ok {
		t.seqByEpochPIDVAddr[epoch][pid] = make(map[uint64]uint64)
	}
	// Use a large baseline to avoid threshold hits in the benchmark.
	t.seqByEpochPIDVAddr[epoch][pid][pageVAddr] = 1
	t.baselineBySeq[1] = 1 << 60

	return t
}

func BenchmarkAutoReleaseObserveAccessNoop(b *testing.B) {
	pid := vm.PID(1)
	pageVAddr := uint64(0x1000)
	now := sim.VTimeInSec(1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = pid
		_ = pageVAddr
		_ = now
	}
}

func BenchmarkAutoReleaseObserveAccess(b *testing.B) {
	tracer := newAccessTracerForBench()
	pid := vm.PID(1)
	pageVAddr := uint64(0x1000)
	now := sim.VTimeInSec(1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tracer.OnPageAccess(pid, pageVAddr, now)
	}
}

func benchmarkProcessDueReleases(b *testing.B, size int) {
	now := sim.VTimeInSec(1)
	b.ReportAllocs()
	b.StopTimer()
	for i := 0; i < b.N; i++ {
		tracer := newEpochPageAllocTracer("", func(uint64) {})
		tracer.enforceOn = true
		tracer.pendingReleaseByEpoch = make(map[int]map[uint64]pendingPageRelease)
		tracer.pendingReleaseByEpoch[2] = make(map[uint64]pendingPageRelease, size)
		for j := 0; j < size; j++ {
			seq := uint64(j + 1)
			tracer.pendingReleaseByEpoch[2][seq] = pendingPageRelease{
				VAddr:     uint64(0x1000 + j*0x1000),
				ReleaseAt: 0,
			}
		}

		b.StartTimer()
		tracer.processDueReleases(now)
		b.StopTimer()
	}
}

func BenchmarkAutoReleaseProcessDueReleases(b *testing.B) {
	sizes := []int{0, 1, 10, 100, 1000}
	for _, size := range sizes {
		b.Run("size_"+strconv.Itoa(size), func(b *testing.B) {
			benchmarkProcessDueReleases(b, size)
		})
	}
}
