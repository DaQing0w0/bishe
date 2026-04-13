package driver

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"sync"

	"github.com/sarchlab/akita/v4/mem/vm"
	"github.com/sarchlab/mgpusim/v4/amd/driver/internal"
)

type epochPageAllocRecord struct {
	Seq      uint64
	Epoch    int
	PID      vm.PID
	Cause    string
	VAddr    uint64
	PAddr    uint64
	PageSize uint64
	DeviceID uint64
	Unified  bool
}

type epochPageAllocTracer struct {
	mu sync.Mutex

	outputDir      string
	nextSeqByEpoch map[int]uint64

	activeEpochByPID map[vm.PID]int
	activePIDCount   map[int]int
	recordsByEpoch   map[int][]epochPageAllocRecord
}

func newEpochPageAllocTracer(outputDir string) *epochPageAllocTracer {
	if outputDir == "" {
		outputDir = "page_alloc_trace"
	}

	return &epochPageAllocTracer{
		outputDir:        outputDir,
		nextSeqByEpoch:   make(map[int]uint64),
		activeEpochByPID: make(map[vm.PID]int),
		activePIDCount:   make(map[int]int),
		recordsByEpoch:   make(map[int][]epochPageAllocRecord),
	}
}

func (t *epochPageAllocTracer) OnPageAllocated(event internal.PageAllocationEvent) {
	t.mu.Lock()
	defer t.mu.Unlock()

	epoch, found := t.activeEpochByPID[event.PID]
	if !found {
		return
	}

	t.nextSeqByEpoch[epoch]++
	t.recordsByEpoch[epoch] = append(t.recordsByEpoch[epoch], epochPageAllocRecord{
		Seq:      t.nextSeqByEpoch[epoch],
		Epoch:    epoch,
		PID:      event.PID,
		Cause:    event.Cause,
		VAddr:    event.VAddr,
		PAddr:    event.PAddr,
		PageSize: event.PageSize,
		DeviceID: event.DeviceID,
		Unified:  event.Unified,
	})
}

func (t *epochPageAllocTracer) beginEpochForPID(pid vm.PID, epoch int) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if oldEpoch, found := t.activeEpochByPID[pid]; found {
		t.activePIDCount[oldEpoch]--
		if t.activePIDCount[oldEpoch] <= 0 {
			delete(t.activePIDCount, oldEpoch)
		}
	}

	t.activeEpochByPID[pid] = epoch
	t.activePIDCount[epoch]++
}

func (t *epochPageAllocTracer) endEpochForPID(pid vm.PID) (epoch int, shouldFlush bool) {
	t.mu.Lock()
	defer t.mu.Unlock()

	epoch, found := t.activeEpochByPID[pid]
	if !found {
		return -1, false
	}

	delete(t.activeEpochByPID, pid)
	t.activePIDCount[epoch]--
	if t.activePIDCount[epoch] > 0 {
		return epoch, false
	}

	delete(t.activePIDCount, epoch)
	return epoch, true
}

func (t *epochPageAllocTracer) flushEpoch(epoch int) error {
	t.mu.Lock()
	records := append([]epochPageAllocRecord(nil), t.recordsByEpoch[epoch]...)
	delete(t.recordsByEpoch, epoch)
	delete(t.nextSeqByEpoch, epoch)
	outDir := t.outputDir
	t.mu.Unlock()

	if len(records) == 0 {
		return nil
	}

	sort.Slice(records, func(i, j int) bool {
		return records[i].Seq < records[j].Seq
	})

	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}

	filePath := filepath.Join(outDir, fmt.Sprintf("epoch_%04d_page_alloc.csv", epoch))
	f, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	if err := w.Write([]string{
		"seq",
		"epoch",
		"pid",
		"cause",
		"vaddr_hex",
		"paddr_hex",
		"page_size",
		"device_id",
		"unified",
	}); err != nil {
		return err
	}

	for _, r := range records {
		if err := w.Write([]string{
			strconv.FormatUint(r.Seq, 10),
			strconv.Itoa(r.Epoch),
			strconv.FormatUint(uint64(r.PID), 10),
			r.Cause,
			fmt.Sprintf("0x%x", r.VAddr),
			fmt.Sprintf("0x%x", r.PAddr),
			strconv.FormatUint(r.PageSize, 10),
			strconv.FormatUint(r.DeviceID, 10),
			strconv.FormatBool(r.Unified),
		}); err != nil {
			return err
		}
	}

	if err := w.Error(); err != nil {
		return err
	}

	return nil
}

func uniquePIDs(contexts []*Context) []vm.PID {
	seen := make(map[vm.PID]bool)
	result := make([]vm.PID, 0, len(contexts))
	for _, c := range contexts {
		if c == nil {
			continue
		}
		if seen[c.pid] {
			continue
		}
		seen[c.pid] = true
		result = append(result, c.pid)
	}
	return result
}

// EnableEpochPageAllocationTrace enables page-allocation tracing and CSV export.
func (d *Driver) EnableEpochPageAllocationTrace(outputDir string) {
	tracer := newEpochPageAllocTracer(outputDir)
	d.pageAllocTracer = tracer
	internal.SetPageAllocationObserver(tracer)
}

// DisableEpochPageAllocationTrace disables page-allocation tracing.
func (d *Driver) DisableEpochPageAllocationTrace() {
	d.pageAllocTracer = nil
	internal.SetPageAllocationObserver(nil)
}

// BeginEpochPageAllocationTrace marks epoch start for all provided contexts.
func (d *Driver) BeginEpochPageAllocationTrace(contexts []*Context, epoch int) {
	if d.pageAllocTracer == nil {
		return
	}

	for _, pid := range uniquePIDs(contexts) {
		d.pageAllocTracer.beginEpochForPID(pid, epoch)
	}
}

// EndEpochPageAllocationTrace marks epoch end for all provided contexts and
// flushes CSV output when no PID remains active in that epoch.
func (d *Driver) EndEpochPageAllocationTrace(contexts []*Context) error {
	if d.pageAllocTracer == nil {
		return nil
	}

	flushedEpochs := make(map[int]bool)
	for _, pid := range uniquePIDs(contexts) {
		epoch, shouldFlush := d.pageAllocTracer.endEpochForPID(pid)
		if !shouldFlush || flushedEpochs[epoch] {
			continue
		}
		if err := d.pageAllocTracer.flushEpoch(epoch); err != nil {
			return err
		}
		flushedEpochs[epoch] = true
	}

	return nil
}
