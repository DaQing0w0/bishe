package internal

import (
	"sync"

	"github.com/sarchlab/akita/v4/mem/vm"
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
}

// PageAllocationObserver receives page-level allocation events.
type PageAllocationObserver interface {
	OnPageAllocated(event PageAllocationEvent)
}

var (
	pageAllocObserverMu sync.RWMutex
	pageAllocObserver   PageAllocationObserver
)

// SetPageAllocationObserver sets a global observer for page allocation events.
// Passing nil disables observation.
func SetPageAllocationObserver(observer PageAllocationObserver) {
	pageAllocObserverMu.Lock()
	defer pageAllocObserverMu.Unlock()

	pageAllocObserver = observer
}

func emitPageAllocated(event PageAllocationEvent) {
	pageAllocObserverMu.RLock()
	observer := pageAllocObserver
	pageAllocObserverMu.RUnlock()

	if observer == nil {
		return
	}

	observer.OnPageAllocated(event)
}
