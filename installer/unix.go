//go:build !windows

package main

import (
	"fmt"
	"os"
	"syscall"
)

func lockState(p string) (*os.File, error) {
	f, e := os.OpenFile(p, os.O_CREATE|os.O_RDWR, 0600)
	if e != nil {
		return nil, e
	}
	if e = syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); e != nil {
		f.Close()
		return nil, fmt.Errorf("another checkpoint switch is active: %w", e)
	}
	return f, nil
}
