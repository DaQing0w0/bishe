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
	"github.com/sarchlab/akita/v4/sim"
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

type pendingPageRelease struct {
	VAddr     uint64
	ReleaseAt sim.VTimeInSec
}

type epochPageAllocTracer struct {
	mu sync.Mutex

	outputDir           string
	allocTraceOn        bool
	dryRunOn            bool
	dryRunOutputDir     string
	enforceOn           bool
	enforceOutputDir    string
	enforceReleaseDelay sim.VTimeInSec
	releasePage         func(vAddr uint64)

	nextSeqByEpoch map[int]uint64

	activeEpochByPID map[vm.PID]int
	activePIDCount   map[int]int
	recordsByEpoch   map[int][]epochPageAllocRecord

	seqByEpochPIDVAddr    map[int]map[vm.PID]map[uint64]uint64
	accessByEpochSeq      map[int]map[uint64]uint64
	allocTimeByEpochSeq   map[int]map[uint64]sim.VTimeInSec
	epochStartTimeByEpoch map[int]sim.VTimeInSec
	baselineBySeq         map[uint64]uint64

	wouldReleaseByEpoch    map[int]map[uint64]bool
	pendingReleaseByEpoch  map[int]map[uint64]pendingPageRelease
	releasedTimeByEpochSeq map[int]map[uint64]sim.VTimeInSec
	unknownSeqByEpoch      map[int]uint64
	postThresholdByEpoch   map[int]map[uint64]uint64
}

func newEpochPageAllocTracer(outputDir string, releasePage func(vAddr uint64)) *epochPageAllocTracer {
	if outputDir == "" {
		outputDir = "page_alloc_trace"
	}

	return &epochPageAllocTracer{
		outputDir:              outputDir,
		allocTraceOn:           true,
		enforceReleaseDelay:    sim.VTimeInSec(1e-6),
		releasePage:            releasePage,
		nextSeqByEpoch:         make(map[int]uint64),
		activeEpochByPID:       make(map[vm.PID]int),
		activePIDCount:         make(map[int]int),
		recordsByEpoch:         make(map[int][]epochPageAllocRecord),
		seqByEpochPIDVAddr:     make(map[int]map[vm.PID]map[uint64]uint64),
		accessByEpochSeq:       make(map[int]map[uint64]uint64),
		allocTimeByEpochSeq:    make(map[int]map[uint64]sim.VTimeInSec),
		epochStartTimeByEpoch:  make(map[int]sim.VTimeInSec),
		baselineBySeq:          make(map[uint64]uint64),
		wouldReleaseByEpoch:    make(map[int]map[uint64]bool),
		pendingReleaseByEpoch:  make(map[int]map[uint64]pendingPageRelease),
		releasedTimeByEpochSeq: make(map[int]map[uint64]sim.VTimeInSec),
		unknownSeqByEpoch:      make(map[int]uint64),
		postThresholdByEpoch:   make(map[int]map[uint64]uint64),
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

func (t *epochPageAllocTracer) processDueReleases(now sim.VTimeInSec) int {
	t.mu.Lock()
	if !t.enforceOn || t.releasePage == nil {
		t.mu.Unlock()
		return 0
	}

	type releaseItem struct {
		epoch int
		seq   uint64
		vAddr uint64
	}
	releases := make([]releaseItem, 0)

	for epoch, pendingBySeq := range t.pendingReleaseByEpoch {
		for seq, pending := range pendingBySeq {
			if now < pending.ReleaseAt {
				continue
			}

			if _, ok := t.releasedTimeByEpochSeq[epoch]; !ok {
				t.releasedTimeByEpochSeq[epoch] = make(map[uint64]sim.VTimeInSec)
			}
			if _, alreadyReleased := t.releasedTimeByEpochSeq[epoch][seq]; alreadyReleased {
				delete(pendingBySeq, seq)
				continue
			}

			t.releasedTimeByEpochSeq[epoch][seq] = now
			releases = append(releases, releaseItem{epoch: epoch, seq: seq, vAddr: pending.VAddr})
			delete(pendingBySeq, seq)
		}

		if len(pendingBySeq) == 0 {
			delete(t.pendingReleaseByEpoch, epoch)
		}
	}
	t.mu.Unlock()

	for _, r := range releases {
		t.releasePage(r.vAddr)
	}

	return len(releases)
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

	if _, ok := t.allocTimeByEpochSeq[epoch]; !ok {
		t.allocTimeByEpochSeq[epoch] = make(map[uint64]sim.VTimeInSec)
	}
	t.allocTimeByEpochSeq[epoch][seq] = event.Time

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

func (t *epochPageAllocTracer) OnPageAccess(pid vm.PID, pageVAddr uint64, now sim.VTimeInSec) {
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

			if t.enforceOn {
				if _, ok := t.releasedTimeByEpochSeq[epoch]; !ok {
					t.releasedTimeByEpochSeq[epoch] = make(map[uint64]sim.VTimeInSec)
				}
				if _, released := t.releasedTimeByEpochSeq[epoch][seq]; !released {
					if _, ok := t.pendingReleaseByEpoch[epoch]; !ok {
						t.pendingReleaseByEpoch[epoch] = make(map[uint64]pendingPageRelease)
					}
					if _, pending := t.pendingReleaseByEpoch[epoch][seq]; !pending {
						t.pendingReleaseByEpoch[epoch][seq] = pendingPageRelease{
							VAddr:     pageVAddr,
							ReleaseAt: now + t.enforceReleaseDelay,
						}
					}
				}
			}
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
		if pendingBySeq, ok := t.pendingReleaseByEpoch[epoch]; ok {
			delete(pendingBySeq, seq)
		}
	}
}

func (t *epochPageAllocTracer) beginEpochForPID(pid vm.PID, epoch int, now sim.VTimeInSec) {
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
	if _, found := t.epochStartTimeByEpoch[epoch]; !found {
		t.epochStartTimeByEpoch[epoch] = now
	}
}

func (t *epochPageAllocTracer) endEpochForPID(pid vm.PID, now sim.VTimeInSec) (
	epoch int,
	shouldFlush bool,
	epochEndTime sim.VTimeInSec,
) {
	t.mu.Lock()
	defer t.mu.Unlock()

	epoch, found := t.activeEpochByPID[pid]
	if !found {
		return -1, false, 0
	}

	delete(t.activeEpochByPID, pid)
	t.activePIDCount[epoch]--
	if t.activePIDCount[epoch] > 0 {
		return epoch, false, 0
	}

	delete(t.activePIDCount, epoch)
	return epoch, true, now
}

func (t *epochPageAllocTracer) flushEpoch(epoch int, epochEndTime sim.VTimeInSec) (int, error) {
	t.mu.Lock()
	records := append([]epochPageAllocRecord(nil), t.recordsByEpoch[epoch]...)
	accessBySeq := make(map[uint64]uint64)
	for seq, c := range t.accessByEpochSeq[epoch] {
		accessBySeq[seq] = c
	}
	allocTimeBySeq := make(map[uint64]sim.VTimeInSec)
	for seq, tm := range t.allocTimeByEpochSeq[epoch] {
		allocTimeBySeq[seq] = tm
	}
	epochStartTime := t.epochStartTimeByEpoch[epoch]
	wouldRelease := make(map[uint64]bool)
	for seq, v := range t.wouldReleaseByEpoch[epoch] {
		wouldRelease[seq] = v
	}
	releasedTimeBySeq := make(map[uint64]sim.VTimeInSec)
	for seq, tm := range t.releasedTimeByEpochSeq[epoch] {
		releasedTimeBySeq[seq] = tm
	}
	released := make(map[uint64]bool)
	for seq := range releasedTimeBySeq {
		released[seq] = true
	}
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
	delete(t.allocTimeByEpochSeq, epoch)
	delete(t.epochStartTimeByEpoch, epoch)
	delete(t.wouldReleaseByEpoch, epoch)
	delete(t.pendingReleaseByEpoch, epoch)
	delete(t.releasedTimeByEpochSeq, epoch)
	delete(t.postThresholdByEpoch, epoch)
	delete(t.unknownSeqByEpoch, epoch)

	if epoch == 1 {
		t.baselineBySeq = make(map[uint64]uint64)
		for seq, c := range accessBySeq {
			t.baselineBySeq[seq] = c
		}
	}

	t.mu.Unlock()

	releaseCount := 0

	if len(records) == 0 {
		return 0, nil
	}

	sort.Slice(records, func(i, j int) bool {
		return records[i].Seq < records[j].Seq
	})

	if allocTraceOn {
		if err := os.MkdirAll(allocOutDir, 0o755); err != nil {
			return 0, err
		}

		filePath := filepath.Join(allocOutDir, fmt.Sprintf("epoch_%04d_page_alloc.csv", epoch))
		f, err := os.Create(filePath)
		if err != nil {
			return 0, err
		}
		defer f.Close()

		w := csv.NewWriter(f)
		defer w.Flush()

		if err := w.Write([]string{
			"seq", "epoch", "pid", "cause", "vaddr_hex", "paddr_hex", "page_size", "device_id", "unified",
		}); err != nil {
			return 0, err
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
				return 0, err
			}
		}

		if err := w.Error(); err != nil {
			return 0, err
		}
	}

	if dryRunOn && epoch >= 2 {
		if err := os.MkdirAll(dryRunOutDir, 0o755); err != nil {
			return 0, err
		}

		filePath := filepath.Join(dryRunOutDir, fmt.Sprintf("epoch_%04d_auto_release_dry_run.csv", epoch))
		f, err := os.Create(filePath)
		if err != nil {
			return 0, err
		}
		defer f.Close()

		w := csv.NewWriter(f)
		defer w.Flush()

		if err := w.Write([]string{
			"seq", "epoch", "baseline_access", "current_access", "would_release", "post_threshold_access",
			"alloc_time", "candidate_release_time", "candidate_lifetime", "epoch_runtime", "candidate_lifetime_epoch_runtime_ratio",
		}); err != nil {
			return 0, err
		}

		epochRuntime := epochEndTime - epochStartTime
		epochRuntimeStr := strconv.FormatFloat(float64(epochRuntime), 'f', 12, 64)

		for _, r := range records {
			baseline := baselineBySeq[r.Seq]
			current := accessBySeq[r.Seq]
			post := postThreshold[r.Seq]
			allocTime := allocTimeBySeq[r.Seq]
			allocTimeStr := strconv.FormatFloat(float64(allocTime), 'f', 12, 64)

			candidateReleaseTimeStr := ""
			candidateLifetimeStr := ""
			candidateRatioStr := ""
			if wouldRelease[r.Seq] {
				candidateReleaseTime := epochEndTime
				candidateLifetime := candidateReleaseTime - allocTime
				candidateReleaseTimeStr = strconv.FormatFloat(float64(candidateReleaseTime), 'f', 12, 64)
				candidateLifetimeStr = strconv.FormatFloat(float64(candidateLifetime), 'f', 12, 64)
				if epochRuntime > 0 {
					candidateRatio := float64(candidateLifetime) / float64(epochRuntime)
					candidateRatioStr = strconv.FormatFloat(candidateRatio, 'f', 12, 64)
				}
			}

			if err := w.Write([]string{
				strconv.FormatUint(r.Seq, 10),
				strconv.Itoa(epoch),
				strconv.FormatUint(baseline, 10),
				strconv.FormatUint(current, 10),
				strconv.FormatBool(wouldRelease[r.Seq]),
				strconv.FormatUint(post, 10),
				allocTimeStr,
				candidateReleaseTimeStr,
				candidateLifetimeStr,
				epochRuntimeStr,
				candidateRatioStr,
			}); err != nil {
				return 0, err
			}
		}

		if err := w.Error(); err != nil {
			return 0, err
		}

		log.Printf("[auto-release-dry-run] epoch=%d would_release=%d unknown_seq_access=%d", epoch, len(wouldRelease), unknownSeq)
	}

	if enforceOn && epoch >= 2 {
		for _, r := range records {
			baseline := baselineBySeq[r.Seq]
			current := accessBySeq[r.Seq]
			post := postThreshold[r.Seq]

			if baseline == 0 || !wouldRelease[r.Seq] || post > 0 || current != baseline || released[r.Seq] {
				continue
			}

			if t.releasePage != nil {
				t.releasePage(r.VAddr)
				released[r.Seq] = true
				releasedTimeBySeq[r.Seq] = epochEndTime
				releaseCount++
			}
		}

		if err := os.MkdirAll(enforceOutDir, 0o755); err != nil {
			return 0, err
		}

		filePath := filepath.Join(enforceOutDir, fmt.Sprintf("epoch_%04d_auto_release_enforce.csv", epoch))
		f, err := os.Create(filePath)
		if err != nil {
			return 0, err
		}
		defer f.Close()

		w := csv.NewWriter(f)
		defer w.Flush()

		if err := w.Write([]string{
			"seq", "epoch", "baseline_access", "current_access", "would_release", "did_release", "post_threshold_access",
			"alloc_time", "release_time", "lifetime", "epoch_runtime", "lifetime_epoch_runtime_ratio",
		}); err != nil {
			return 0, err
		}

		epochRuntime := epochEndTime - epochStartTime
		epochRuntimeStr := strconv.FormatFloat(float64(epochRuntime), 'f', 12, 64)

		for _, r := range records {
			baseline := baselineBySeq[r.Seq]
			current := accessBySeq[r.Seq]
			post := postThreshold[r.Seq]
			allocTime := allocTimeBySeq[r.Seq]
			allocTimeStr := strconv.FormatFloat(float64(allocTime), 'f', 12, 64)

			releaseTimeStr := ""
			lifetimeStr := ""
			ratioStr := ""
			if released[r.Seq] {
				releaseTime := releasedTimeBySeq[r.Seq]
				lifetime := releaseTime - allocTime
				releaseTimeStr = strconv.FormatFloat(float64(releaseTime), 'f', 12, 64)
				lifetimeStr = strconv.FormatFloat(float64(lifetime), 'f', 12, 64)
				if epochRuntime > 0 {
					ratio := float64(lifetime) / float64(epochRuntime)
					ratioStr = strconv.FormatFloat(ratio, 'f', 12, 64)
				}
			}

			if err := w.Write([]string{
				strconv.FormatUint(r.Seq, 10),
				strconv.Itoa(epoch),
				strconv.FormatUint(baseline, 10),
				strconv.FormatUint(current, 10),
				strconv.FormatBool(wouldRelease[r.Seq]),
				strconv.FormatBool(released[r.Seq]),
				strconv.FormatUint(post, 10),
				allocTimeStr,
				releaseTimeStr,
				lifetimeStr,
				epochRuntimeStr,
				ratioStr,
			}); err != nil {
				return 0, err
			}
		}

		if err := w.Error(); err != nil {
			return 0, err
		}

		log.Printf("[auto-release-enforce] epoch=%d released=%d unknown_seq_access=%d", epoch, len(released), unknownSeq)
	}

	return releaseCount, nil
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
		internal.SetPageAllocationTimeSource(func() sim.VTimeInSec {
			return d.Engine.CurrentTime()
		})
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
		internal.SetPageAllocationTimeSource(nil)
	}
}

// EnableAutoPageReleaseDryRun enables online auto-release decision dry-run.
func (d *Driver) EnableAutoPageReleaseDryRun(outputDir string) {
	if d.pageAllocTracer == nil {
		d.pageAllocTracer = newEpochPageAllocTracer("", d.memAllocator.RemovePage)
		internal.SetPageAllocationObserver(d.pageAllocTracer)
		internal.SetPageAllocationTimeSource(func() sim.VTimeInSec {
			return d.Engine.CurrentTime()
		})
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
		internal.SetPageAllocationTimeSource(nil)
	}
}

// EnableAutoPageReleaseEnforce enables online threshold-based page release.
func (d *Driver) EnableAutoPageReleaseEnforce(outputDir string) {
	if d.pageAllocTracer == nil {
		d.pageAllocTracer = newEpochPageAllocTracer("", d.memAllocator.RemovePage)
		internal.SetPageAllocationObserver(d.pageAllocTracer)
		internal.SetPageAllocationTimeSource(func() sim.VTimeInSec {
			return d.Engine.CurrentTime()
		})
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
		internal.SetPageAllocationTimeSource(nil)
	}
}

// ObservePageAccessForAutoRelease accepts per-page access events for auto-release modes.
func (d *Driver) ObservePageAccessForAutoRelease(pid vm.PID, pageVAddr uint64) {
	if d.pageAllocTracer == nil {
		return
	}

	d.pageAllocTracer.OnPageAccess(pid, pageVAddr, d.Engine.CurrentTime())
	d.addAutoReleaseAccessOverhead()
}

// BeginEpochPageAllocationTrace marks epoch start for all provided contexts.
func (d *Driver) BeginEpochPageAllocationTrace(contexts []*Context, epoch int) {
	if d.pageAllocTracer == nil {
		return
	}
	now := d.Engine.CurrentTime()

	for _, pid := range uniquePIDs(contexts) {
		d.pageAllocTracer.beginEpochForPID(pid, epoch, now)
	}
}

// EndEpochPageAllocationTrace marks epoch end for all provided contexts and
// flushes CSV output when no PID remains active in that epoch.
func (d *Driver) EndEpochPageAllocationTrace(contexts []*Context) error {
	if d.pageAllocTracer == nil {
		return nil
	}

	flushedEpochs := make(map[int]bool)
	now := d.Engine.CurrentTime()
	for _, pid := range uniquePIDs(contexts) {
		epoch, shouldFlush, epochEndTime := d.pageAllocTracer.endEpochForPID(pid, now)
		if !shouldFlush || flushedEpochs[epoch] {
			continue
		}
		released, err := d.pageAllocTracer.flushEpoch(epoch, epochEndTime)
		if err != nil {
			return err
		}
		if released > 0 {
			d.addAutoReleaseReleaseOverhead(released)
		}
		flushedEpochs[epoch] = true
	}

	return nil
}
