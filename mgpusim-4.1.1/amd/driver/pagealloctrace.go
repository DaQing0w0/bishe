package driver

import (
	"encoding/csv"
	"fmt"
	"log"
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

	outputDir        string
	allocTraceOn     bool
	dryRunOn         bool
	dryRunOutputDir  string
	enforceOn        bool
	enforceOutputDir string
	releasePage      func(vAddr uint64)

	nextSeqByEpoch map[int]uint64

	activeEpochByPID map[vm.PID]int
	activePIDCount   map[int]int
	recordsByEpoch   map[int][]epochPageAllocRecord

	seqByEpochPIDVAddr map[int]map[vm.PID]map[uint64]uint64
	accessByEpochSeq   map[int]map[uint64]uint64
	baselineBySeq      map[uint64]uint64

	wouldReleaseByEpoch  map[int]map[uint64]bool
	unknownSeqByEpoch    map[int]uint64
	postThresholdByEpoch map[int]map[uint64]uint64
}

func newEpochPageAllocTracer(outputDir string, releasePage func(vAddr uint64)) *epochPageAllocTracer {
	if outputDir == "" {
		outputDir = "page_alloc_trace"
	}

	return &epochPageAllocTracer{
		outputDir:            outputDir,
		allocTraceOn:         true,
		releasePage:          releasePage,
		nextSeqByEpoch:       make(map[int]uint64),
		activeEpochByPID:     make(map[vm.PID]int),
		activePIDCount:       make(map[int]int),
		recordsByEpoch:       make(map[int][]epochPageAllocRecord),
		seqByEpochPIDVAddr:   make(map[int]map[vm.PID]map[uint64]uint64),
		accessByEpochSeq:     make(map[int]map[uint64]uint64),
		baselineBySeq:        make(map[uint64]uint64),
		wouldReleaseByEpoch:  make(map[int]map[uint64]bool),
		unknownSeqByEpoch:    make(map[int]uint64),
		postThresholdByEpoch: make(map[int]map[uint64]uint64),
	}
}

func (t *epochPageAllocTracer) setAllocTraceOutputDir(outputDir string) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if outputDir == "" {
		outputDir = "page_alloc_trace"
	}
	t.outputDir = outputDir
	t.allocTraceOn = true
}

func (t *epochPageAllocTracer) disableAllocTrace() {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.allocTraceOn = false
}

func (t *epochPageAllocTracer) enableAutoReleaseDryRun(outputDir string) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if outputDir == "" {
		outputDir = "page_auto_release_dry_run"
	}
	t.dryRunOutputDir = outputDir
	t.dryRunOn = true
}

func (t *epochPageAllocTracer) disableAutoReleaseDryRun() {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.dryRunOn = false
}

func (t *epochPageAllocTracer) enableAutoReleaseEnforce(outputDir string) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if outputDir == "" {
		outputDir = "page_auto_release_enforce"
	}
	t.enforceOutputDir = outputDir
	t.enforceOn = true
}

func (t *epochPageAllocTracer) disableAutoReleaseEnforce() {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.enforceOn = false
}

func (t *epochPageAllocTracer) isIdle() bool {
	t.mu.Lock()
	defer t.mu.Unlock()

	return !t.allocTraceOn && !t.dryRunOn && !t.enforceOn
}

func (t *epochPageAllocTracer) OnPageAllocated(event internal.PageAllocationEvent) {
	t.mu.Lock()
	defer t.mu.Unlock()

	epoch, found := t.activeEpochByPID[event.PID]
	if !found {
		return
	}

	t.nextSeqByEpoch[epoch]++
	seq := t.nextSeqByEpoch[epoch]
	if _, ok := t.seqByEpochPIDVAddr[epoch]; !ok {
		t.seqByEpochPIDVAddr[epoch] = make(map[vm.PID]map[uint64]uint64)
	}
	if _, ok := t.seqByEpochPIDVAddr[epoch][event.PID]; !ok {
		t.seqByEpochPIDVAddr[epoch][event.PID] = make(map[uint64]uint64)
	}
	t.seqByEpochPIDVAddr[epoch][event.PID][event.VAddr] = seq

	t.recordsByEpoch[epoch] = append(t.recordsByEpoch[epoch], epochPageAllocRecord{
		Seq:      seq,
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

func (t *epochPageAllocTracer) OnPageAccess(pid vm.PID, pageVAddr uint64) {
	t.mu.Lock()
	defer t.mu.Unlock()

	epoch, found := t.activeEpochByPID[pid]
	if !found {
		return
	}

	pidMap, ok := t.seqByEpochPIDVAddr[epoch][pid]
	if !ok {
		return
	}

	seq, ok := pidMap[pageVAddr]
	if !ok {
		return
	}

	if _, ok := t.accessByEpochSeq[epoch]; !ok {
		t.accessByEpochSeq[epoch] = make(map[uint64]uint64)
	}
	t.accessByEpochSeq[epoch][seq]++

	if epoch < 2 {
		return
	}

	baseline, ok := t.baselineBySeq[seq]
	if !ok {
		t.unknownSeqByEpoch[epoch]++
		return
	}

	current := t.accessByEpochSeq[epoch][seq]
	if current == baseline {
		if _, ok := t.wouldReleaseByEpoch[epoch]; !ok {
			t.wouldReleaseByEpoch[epoch] = make(map[uint64]bool)
		}
		if baseline > 0 {
			t.wouldReleaseByEpoch[epoch][seq] = true
		}

		if !t.dryRunOn {
			return
		}
		return
	}

	if current > baseline {
		if _, ok := t.postThresholdByEpoch[epoch]; !ok {
			t.postThresholdByEpoch[epoch] = make(map[uint64]uint64)
		}
		t.postThresholdByEpoch[epoch][seq]++
	}
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
	accessBySeq := make(map[uint64]uint64)
	for seq, c := range t.accessByEpochSeq[epoch] {
		accessBySeq[seq] = c
	}
	wouldRelease := make(map[uint64]bool)
	for seq, v := range t.wouldReleaseByEpoch[epoch] {
		wouldRelease[seq] = v
	}
	released := make(map[uint64]bool)
	postThreshold := make(map[uint64]uint64)
	for seq, c := range t.postThresholdByEpoch[epoch] {
		postThreshold[seq] = c
	}
	unknownSeq := t.unknownSeqByEpoch[epoch]
	baselineBySeq := make(map[uint64]uint64)
	for seq, c := range t.baselineBySeq {
		baselineBySeq[seq] = c
	}
	allocTraceOn := t.allocTraceOn
	dryRunOn := t.dryRunOn
	enforceOn := t.enforceOn
	allocOutDir := t.outputDir
	dryRunOutDir := t.dryRunOutputDir
	enforceOutDir := t.enforceOutputDir

	delete(t.recordsByEpoch, epoch)
	delete(t.nextSeqByEpoch, epoch)
	delete(t.seqByEpochPIDVAddr, epoch)
	delete(t.accessByEpochSeq, epoch)
	delete(t.wouldReleaseByEpoch, epoch)
	delete(t.postThresholdByEpoch, epoch)
	delete(t.unknownSeqByEpoch, epoch)

	if epoch == 1 {
		t.baselineBySeq = make(map[uint64]uint64)
		for seq, c := range accessBySeq {
			t.baselineBySeq[seq] = c
		}
	}

	t.mu.Unlock()

	if len(records) == 0 {
		return nil
	}

	sort.Slice(records, func(i, j int) bool {
		return records[i].Seq < records[j].Seq
	})

	if allocTraceOn {
		if err := os.MkdirAll(allocOutDir, 0o755); err != nil {
			return err
		}

		filePath := filepath.Join(allocOutDir, fmt.Sprintf("epoch_%04d_page_alloc.csv", epoch))
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
	}

	if dryRunOn && epoch >= 2 {
		if err := os.MkdirAll(dryRunOutDir, 0o755); err != nil {
			return err
		}

		filePath := filepath.Join(dryRunOutDir, fmt.Sprintf("epoch_%04d_auto_release_dry_run.csv", epoch))
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
			"baseline_access",
			"current_access",
			"would_release",
			"post_threshold_access",
		}); err != nil {
			return err
		}

		for _, r := range records {
			baseline := baselineBySeq[r.Seq]
			current := accessBySeq[r.Seq]
			post := postThreshold[r.Seq]
			if err := w.Write([]string{
				strconv.FormatUint(r.Seq, 10),
				strconv.Itoa(epoch),
				strconv.FormatUint(baseline, 10),
				strconv.FormatUint(current, 10),
				strconv.FormatBool(wouldRelease[r.Seq]),
				strconv.FormatUint(post, 10),
			}); err != nil {
				return err
			}
		}

		if err := w.Error(); err != nil {
			return err
		}

		log.Printf("[auto-release-dry-run] epoch=%d would_release=%d unknown_seq_access=%d", epoch, len(wouldRelease), unknownSeq)
	}

	if enforceOn && epoch >= 2 {
		for _, r := range records {
			baseline := baselineBySeq[r.Seq]
			current := accessBySeq[r.Seq]
			post := postThreshold[r.Seq]

			if baseline == 0 || !wouldRelease[r.Seq] || post > 0 || current != baseline {
				continue
			}

			if t.releasePage != nil {
				t.releasePage(r.VAddr)
				released[r.Seq] = true
			}
		}

		if err := os.MkdirAll(enforceOutDir, 0o755); err != nil {
			return err
		}

		filePath := filepath.Join(enforceOutDir, fmt.Sprintf("epoch_%04d_auto_release_enforce.csv", epoch))
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
			"baseline_access",
			"current_access",
			"would_release",
			"did_release",
			"post_threshold_access",
		}); err != nil {
			return err
		}

		for _, r := range records {
			baseline := baselineBySeq[r.Seq]
			current := accessBySeq[r.Seq]
			post := postThreshold[r.Seq]
			if err := w.Write([]string{
				strconv.FormatUint(r.Seq, 10),
				strconv.Itoa(epoch),
				strconv.FormatUint(baseline, 10),
				strconv.FormatUint(current, 10),
				strconv.FormatBool(wouldRelease[r.Seq]),
				strconv.FormatBool(released[r.Seq]),
				strconv.FormatUint(post, 10),
			}); err != nil {
				return err
			}
		}

		if err := w.Error(); err != nil {
			return err
		}

		log.Printf("[auto-release-enforce] epoch=%d released=%d unknown_seq_access=%d", epoch, len(released), unknownSeq)
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
	if d.pageAllocTracer == nil {
		d.pageAllocTracer = newEpochPageAllocTracer(outputDir, d.memAllocator.RemovePage)
		internal.SetPageAllocationObserver(d.pageAllocTracer)
		return
	}

	d.pageAllocTracer.setAllocTraceOutputDir(outputDir)
}

// DisableEpochPageAllocationTrace disables page-allocation tracing.
func (d *Driver) DisableEpochPageAllocationTrace() {
	if d.pageAllocTracer == nil {
		return
	}

	d.pageAllocTracer.disableAllocTrace()
	if d.pageAllocTracer.isIdle() {
		d.pageAllocTracer = nil
		internal.SetPageAllocationObserver(nil)
	}
}

// EnableAutoPageReleaseDryRun enables online auto-release decision dry-run.
func (d *Driver) EnableAutoPageReleaseDryRun(outputDir string) {
	if d.pageAllocTracer == nil {
		d.pageAllocTracer = newEpochPageAllocTracer("", d.memAllocator.RemovePage)
		internal.SetPageAllocationObserver(d.pageAllocTracer)
	}

	d.pageAllocTracer.enableAutoReleaseDryRun(outputDir)
}

// DisableAutoPageReleaseDryRun disables online auto-release decision dry-run.
func (d *Driver) DisableAutoPageReleaseDryRun() {
	if d.pageAllocTracer == nil {
		return
	}

	d.pageAllocTracer.disableAutoReleaseDryRun()
	if d.pageAllocTracer.isIdle() {
		d.pageAllocTracer = nil
		internal.SetPageAllocationObserver(nil)
	}
}

// EnableAutoPageReleaseEnforce enables online threshold-based page release.
func (d *Driver) EnableAutoPageReleaseEnforce(outputDir string) {
	if d.pageAllocTracer == nil {
		d.pageAllocTracer = newEpochPageAllocTracer("", d.memAllocator.RemovePage)
		internal.SetPageAllocationObserver(d.pageAllocTracer)
	}

	d.pageAllocTracer.enableAutoReleaseEnforce(outputDir)
}

// DisableAutoPageReleaseEnforce disables online threshold-based page release.
func (d *Driver) DisableAutoPageReleaseEnforce() {
	if d.pageAllocTracer == nil {
		return
	}

	d.pageAllocTracer.disableAutoReleaseEnforce()
	if d.pageAllocTracer.isIdle() {
		d.pageAllocTracer = nil
		internal.SetPageAllocationObserver(nil)
	}
}

// ObservePageAccessForAutoRelease accepts per-page access events for auto-release modes.
func (d *Driver) ObservePageAccessForAutoRelease(pid vm.PID, pageVAddr uint64) {
	if d.pageAllocTracer == nil {
		return
	}

	d.pageAllocTracer.OnPageAccess(pid, pageVAddr)
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
