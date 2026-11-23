package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func must(t *testing.T, e error) {
	t.Helper()
	if e != nil {
		t.Fatal(e)
	}
}
func fixture(t *testing.T, dir, id, body string) checkpoint {
	t.Helper()
	p := "vllm-ascend-triage/SKILL.md"
	raw, e := json.Marshal([]record{{p, "100644", sum([]byte(body)), int64(len(body))}})
	must(t, e)
	c := checkpoint{strings.Repeat(id, 40), 1, sum(raw)}
	var buf bytes.Buffer
	z := gzip.NewWriter(&buf)
	a := tar.NewWriter(z)
	for _, v := range []struct {
		p string
		b []byte
	}{{"skills/" + p, []byte(body)}, {"files.json", raw}} {
		must(t, a.WriteHeader(&tar.Header{Name: "a-snapshot/" + v.p, Mode: 0644, Size: int64(len(v.b)), Typeflag: tar.TypeReg}))
		_, e = a.Write(v.b)
		must(t, e)
	}
	must(t, a.Close())
	must(t, z.Close())
	must(t, os.WriteFile(filepath.Join(dir, c.Commit+".tar.gz"), buf.Bytes(), 0600))
	return c
}
func TestSwitchFailureOwnershipAndEmptyBaseline(t *testing.T) {
	o := options{offline: t.TempDir(), userDir: filepath.Join(t.TempDir(), "profile with spaces")}
	oldSHA := indexSHA
	defer func() { indexSHA = oldSHA }()
	c1, c2 := fixture(t, o.offline, "a", "first"), fixture(t, o.offline, "b", "second")
	idx := index{"csi-training-checkpoints/v1", map[string]checkpoint{"K1": c1, "K76": c2, "legacy-K20": c1}, map[string]string{"2026-11-18-k20": "legacy-K20"}}
	data, e := json.Marshal(idx)
	must(t, e)
	indexSHA = sum(data)
	must(t, os.WriteFile(filepath.Join(o.offline, "manifest.json"), data, 0600))
	must(t, install(o, "K76"))
	active := filepath.Join(o.userDir, ".agents", "skills", name)
	entry := filepath.Join(active, "SKILL.md")
	before, e := os.ReadFile(entry)
	must(t, e)
	if !bytes.Contains(before, []byte("K76")) {
		t.Fatal("wrong active checkpoint")
	}
	archivePath := filepath.Join(o.offline, c1.Commit+".tar.gz")
	good, e := os.ReadFile(archivePath)
	must(t, e)
	must(t, os.WriteFile(archivePath, []byte("truncated"), 0600))
	if install(o, "K1") == nil {
		t.Fatal("accepted incomplete download")
	}
	after, e := os.ReadFile(entry)
	must(t, e)
	if !bytes.Equal(before, after) {
		t.Fatal("failed switch changed entry")
	}
	must(t, os.WriteFile(archivePath, good, 0600))
	must(t, install(o, "1"))
	must(t, install(o, "K1"))
	must(t, install(o, "2026-11-18-k20"))
	must(t, install(o, "K0"))
	must(t, install(o, "0"))
	if _, e = os.Lstat(active); !os.IsNotExist(e) {
		t.Fatal("K0 kept a discoverable entry")
	}
	must(t, os.Mkdir(active, 0700))
	must(t, os.WriteFile(entry, []byte("user-owned"), 0600))
	for _, choice := range []string{"K76", "K0"} {
		if install(o, choice) == nil {
			t.Fatal("accepted unrelated skill")
		}
	}
	after, e = os.ReadFile(entry)
	must(t, e)
	if string(after) != "user-owned" {
		t.Fatal("user skill changed")
	}
}
func TestEditedOriginalOrWrapperIsPreserved(t *testing.T) {
	dir := t.TempDir()
	c := fixture(t, dir, "c", "test")
	b, e := os.ReadFile(filepath.Join(dir, c.Commit+".tar.gz"))
	must(t, e)
	stage := t.TempDir()
	must(t, archive(b, c, stage))
	f := filepath.Join(stage, "skills", "vllm-ascend-triage", "SKILL.md")
	must(t, os.WriteFile(f, []byte("edit"), 0644))
	if _, e = checkSnapshot(stage, c); e == nil {
		t.Fatal("accepted edited snapshot")
	}
	a := t.TempDir()
	key := []byte("fixture")
	signedEntry := signed([]byte("body"), key)
	must(t, os.WriteFile(filepath.Join(a, "SKILL.md"), signedEntry, 0644))
	ok, e := owned(a, key)
	must(t, e)
	if !ok {
		t.Fatal("lost ownership")
	}
	must(t, os.WriteFile(filepath.Join(a, "SKILL.md"), append(signedEntry, '!'), 0644))
	if _, e := owned(a, key); e == nil {
		t.Fatal("did not protect user's edit")
	}
}
