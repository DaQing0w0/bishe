package internal

import (
	"sync"

	"github.com/sarchlab/akita/v4/mem/vm"
	"github.com/sarchlab/akita/v4/sim"
)

// PageAllocationEvent represents one page allocation in allocator order.
type PageAllocationEvent struct {
	PID      vm.PID
	VAddr    uint64
	PAddr    uint64
	PageSize uint64
	DeviceID uint64
	Unified  bool
	Cause    string
	Time     sim.VTimeInSec
}

// PageAllocationObserver receives page-level allocation events.
type PageAllocationObserver interface {
	OnPageAllocated(event PageAllocationEvent)
}

var (
	pageAllocObserverMu sync.RWMutex
	pageAllocObserver   PageAllocationObserver
	pageAllocTimeSource func() sim.VTimeInSec
)

// SetPageAllocationObserver sets a global observer for page allocation events.
// Passing nil disables observation.
func SetPageAllocationObserver(observer PageAllocationObserver) {
	pageAllocObserverMu.Lock()
	defer pageAllocObserverMu.Unlock()

	pageAllocObserver = observer
}

// SetPageAllocationTimeSource sets the virtual time source for page allocation events.
// Passing nil disables timestamp injection.
func SetPageAllocationTimeSource(source func() sim.VTimeInSec) {
	pageAllocObserverMu.Lock()
	defer pageAllocObserverMu.Unlock()

	pageAllocTimeSource = source
}

func emitPageAllocated(event PageAllocationEvent) {
	pageAllocObserverMu.RLock()
	observer := pageAllocObserver
	timeSource := pageAllocTimeSource
	pageAllocObserverMu.RUnlock()

	if observer == nil {
		return
	}

	if timeSource != nil {
		event.Time = timeSource()
	}

	observer.OnPageAllocated(event)
}
