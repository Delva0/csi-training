// Command csi-install selects one published CSI checkpoint for a local Codex user.
package main

import (
	"archive/tar"
	"bufio"
	"bytes"
	"compress/gzip"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
)

const tag = "2026-11-22-mainline"
const name = "csi-vllm-ascend"
const manifestURL = "https://raw.githubusercontent.com/Delva0/csi-training/" + tag + "/checkpoints/" + tag + "/manifest.json"
const marker = "\n<!-- csi-training-owner "

var indexSHA string // Set from the verified release inputs at build time.
var client = &http.Client{Timeout: 2 * time.Minute}

type checkpoint struct {
	Commit    string `json:"commit"`
	Count     int    `json:"file_count"`
	Inventory string `json:"inventory_sha256"`
}
type index struct {
	Schema      string                `json:"schema"`
	Checkpoints map[string]checkpoint `json:"checkpoints"`
	Aliases     map[string]string     `json:"aliases"`
}
type record struct {
	Path string `json:"path"`
	Mode string `json:"mode"`
	SHA  string `json:"sha256"`
	Size int64  `json:"size"`
}
type options struct {
	offline, userDir string
}

func sum(b []byte) string        { return fmt.Sprintf("%x", sha256.Sum256(b)) }
func hexID(s string, n int) bool { _, err := hex.DecodeString(s); return len(s) == n && err == nil }
func readLimit(r io.Reader, max int64) ([]byte, error) {
	b, err := io.ReadAll(io.LimitReader(r, max+1))
	if err == nil && int64(len(b)) > max {
		err = errors.New("download exceeds release size limit")
	}
	return b, err
}
func fetch(o options, file, url string, max int64) ([]byte, error) {
	if o.offline != "" {
		f, e := os.Open(filepath.Join(o.offline, file))
		if e != nil {
			return nil, e
		}
		defer f.Close()
		return readLimit(f, max)
	}
	resp, e := client.Get(url)
	if e != nil {
		return nil, e
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("%s: HTTP %s", url, resp.Status)
	}
	return readLimit(resp.Body, max)
}
func verifiedIndex(o options) (index, error) {
	var data index
	b, e := fetch(o, "manifest.json", manifestURL, 1<<20)
	if e != nil {
		return data, e
	}
	if !hexID(indexSHA, 64) || sum(b) != indexSHA {
		return data, errors.New("release manifest SHA-256 mismatch")
	}
	e = json.Unmarshal(b, &data)
	if e == nil && data.Schema != "csi-training-checkpoints/v1" {
		e = errors.New("unexpected release manifest")
	}
	return data, e
}
func resolve(data index, input string) (string, checkpoint, error) {
	input = strings.TrimSpace(input)
	if num, e := strconv.Atoi(input); e == nil {
		input = "K" + strconv.Itoa(num)
	}
	for k, a := range data.Aliases {
		if strings.EqualFold(k, input) {
			input = a
			break
		}
	}
	for k, c := range data.Checkpoints {
		if strings.EqualFold(k, input) && hexID(c.Commit, 40) && hexID(c.Inventory, 64) && c.Count >= 0 {
			return k, c, nil
		}
	}
	return "", checkpoint{}, fmt.Errorf("unknown checkpoint %q (K0-K76, legacy-K20, flat-K20)", input)
}
func safePath(p string) bool {
	return p != "" && p != "." && path.Clean(p) == p && !strings.ContainsAny(p, "\\:\x00") && filepath.IsLocal(filepath.FromSlash(p))
}
func inventory(raw []byte, c checkpoint) ([]record, error) {
	if sum(raw) != c.Inventory {
		return nil, errors.New("checkpoint inventory SHA-256 mismatch")
	}
	var rows []record
	if e := json.Unmarshal(raw, &rows); e != nil {
		return nil, e
	}
	if len(rows) != c.Count {
		return nil, errors.New("unexpected checkpoint file count")
	}
	seen := map[string]bool{}
	for _, r := range rows {
		key := strings.ToLower(r.Path)
		if !safePath(r.Path) || seen[key] || !hexID(r.SHA, 64) || r.Size < 0 || r.Size > 64<<20 || (r.Mode != "100644" && r.Mode != "100755") {
			return nil, errors.New("invalid checkpoint file inventory")
		}
		seen[key] = true
	}
	return rows, nil
}
func archive(raw []byte, c checkpoint, dir string) error {
	z, e := gzip.NewReader(bytes.NewReader(raw))
	if e != nil {
		return e
	}
	defer z.Close()
	t := tar.NewReader(z)
	seen := map[string]bool{}
	var meta []byte
	var written int64
	var top string
	for {
		h, e := t.Next()
		if e == io.EOF {
			break
		}
		if e != nil {
			return e
		}
		if h.Typeflag == tar.TypeDir || h.Typeflag == tar.TypeXGlobalHeader {
			continue
		}
		a, p, ok := strings.Cut(h.Name, "/")
		if !ok || !safePath(a) || !safePath(p) || h.Typeflag != tar.TypeReg {
			return errors.New("invalid checkpoint archive entry")
		}
		if top == "" {
			top = a
		}
		if top != a {
			return errors.New("multiple checkpoint archive roots")
		}
		if p != "files.json" && !strings.HasPrefix(p, "skills/") {
			continue
		}
		if seen[p] || h.Size < 0 || h.Size > 64<<20 || written+h.Size > 128<<20 {
			return errors.New("duplicate or oversized checkpoint entry")
		}
		seen[p] = true
		written += h.Size
		if p == "files.json" {
			meta, e = readLimit(t, 4<<20)
			if e != nil {
				return e
			}
			continue
		}
		target := filepath.Join(dir, filepath.FromSlash(p))
		if e = os.MkdirAll(filepath.Dir(target), 0755); e != nil {
			return e
		}
		mode := os.FileMode(0644)
		if h.Mode&0111 != 0 {
			mode = 0755
		}
		f, e := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, mode)
		if e != nil {
			return e
		}
		_, e = io.Copy(f, t)
		closeErr := f.Close()
		if e != nil {
			return e
		}
		if closeErr != nil {
			return closeErr
		}
		if runtime.GOOS != "windows" {
			if e = os.Chmod(target, mode); e != nil {
				return e
			}
		}
	}
	if _, e = inventory(meta, c); e != nil {
		return e
	}
	if e = os.WriteFile(filepath.Join(dir, "files.json"), meta, 0644); e != nil {
		return e
	}
	_, e = checkSnapshot(dir, c)
	return e
}
func checkSnapshot(dir string, c checkpoint) ([]string, error) {
	meta, e := os.ReadFile(filepath.Join(dir, "files.json"))
	if e != nil {
		return nil, e
	}
	rows, e := inventory(meta, c)
	if e != nil {
		return nil, e
	}
	base := filepath.Join(dir, "skills")
	var roots []string
	known := map[string]record{}
	for _, r := range rows {
		known[r.Path] = r
	}
	e = filepath.WalkDir(base, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		rel, e := filepath.Rel(base, p)
		if e != nil {
			return e
		}
		rel = filepath.ToSlash(rel)
		r, ok := known[rel]
		if !ok || !d.Type().IsRegular() {
			return fmt.Errorf("unregistered checkpoint file: %s", rel)
		}
		st, e := d.Info()
		if e != nil {
			return e
		}
		if st.Size() != r.Size || (runtime.GOOS != "windows" && ((st.Mode()&0111 != 0) != (r.Mode == "100755"))) {
			return fmt.Errorf("checkpoint size or mode mismatch: %s", rel)
		}
		b, e := os.ReadFile(p)
		if e != nil {
			return e
		}
		if sum(b) != r.SHA {
			return fmt.Errorf("checkpoint SHA-256 mismatch: %s", rel)
		}
		delete(known, rel)
		if strings.Count(rel, "/") == 1 && path.Base(rel) == "SKILL.md" {
			roots = append(roots, rel)
		}
		return nil
	})
	if e != nil {
		return nil, e
	}
	if len(known) != 0 || len(roots) == 0 {
		return nil, errors.New("incomplete checkpoint skills")
	}
	sort.Slice(roots, func(i, j int) bool {
		a, b := strings.HasPrefix(roots[i], "vllm-ascend"), strings.HasPrefix(roots[j], "vllm-ascend")
		if a != b {
			return a
		}
		return roots[i] < roots[j]
	})
	return roots, nil
}
func snapshot(o options, state string, c checkpoint) (string, []string, error) {
	dir := filepath.Join(state, "snapshots", c.Commit)
	parent := filepath.Dir(dir)
	if st, e := os.Lstat(dir); e == nil {
		if !st.IsDir() {
			return "", nil, errors.New("unexpected snapshot location")
		}
		r, e := checkSnapshot(dir, c)
		return dir, r, e
	} else if !errors.Is(e, os.ErrNotExist) {
		return "", nil, e
	}
	if e := os.MkdirAll(parent, 0700); e != nil {
		return "", nil, e
	}
	stage, e := os.MkdirTemp(parent, ".incoming-")
	if e != nil {
		return "", nil, e
	}
	defer os.RemoveAll(stage)
	b, e := fetch(o, c.Commit+".tar.gz", "https://codeload.github.com/Delva0/csi-training/tar.gz/"+c.Commit, 64<<20)
	if e != nil {
		return "", nil, e
	}
	if e = archive(b, c, stage); e != nil {
		return "", nil, e
	}
	roots, e := checkSnapshot(stage, c)
	if e != nil {
		return "", nil, e
	}
	if e = os.Rename(stage, dir); e != nil {
		return "", nil, e
	}
	return dir, roots, nil
}
func wrapper(id, dir string, roots []string) []byte {
	var b strings.Builder
	fmt.Fprintf(&b, "---\nname: %s\ndescription: Triage vLLM-Ascend and Ascend NPU deployment, inference, and runtime failures using logs, source code, and the selected CSI knowledge checkpoint.\n---\n\n", name)
	fmt.Fprintf(&b, "# vLLM-Ascend 故障排查 · %s\n\n", id)
	b.WriteString("先用本机文件读取工具或终端打开下面的主入口。训练用的工具和挂载路径可以这样在桌面查找。\n\n")
	b.WriteString("- `skill_view`、`skills_list` 对应读取下列 SKILL.md 及其同目录下的 references/、scripts/。用本机文件读取、搜索或终端打开相对路径。\n")
	b.WriteString("- `/workspace/repos` 和 `SOURCES.json` 是训练机的源码清单。先在当前工作区找相关仓库，用 git 核对实际版本。缺源码时使用已连接的 GitHub 工具、可用的 gh，或向用户索取相关版本与代码。\n")
	b.WriteString("- `/workspace/scratch` 对应当前有写权限的任务目录。运行随附脚本前先读脚本并检查解释器、参数和输入路径。Windows 上的 Bash 脚本可在已配置的 WSL 中运行。没有对应工具、日志或源码时说明缺少的证据。\n\n")
	for i, p := range roots {
		label := "主入口"
		if i > 0 {
			label = "同版本的相关技能"
		}
		fmt.Fprintf(&b, "- %s：`%s`\n", label, filepath.Join(dir, "skills", filepath.FromSlash(p)))
	}
	return []byte(b.String())
}
func signed(body, key []byte) []byte {
	h := hmac.New(sha256.New, key)
	h.Write(body)
	return append(body, []byte(marker+hex.EncodeToString(h.Sum(nil))+" -->\n")...)
}
func owned(active string, key []byte) (bool, error) {
	st, e := os.Lstat(active)
	if errors.Is(e, os.ErrNotExist) {
		return false, nil
	}
	if e != nil {
		return false, e
	}
	if !st.IsDir() {
		return false, errors.New("existing target is not an owned skill directory")
	}
	files, e := os.ReadDir(active)
	if e != nil {
		return false, e
	}
	if len(files) != 1 || files[0].Name() != "SKILL.md" || !files[0].Type().IsRegular() {
		return false, fmt.Errorf("preserving another or modified skill at %s", active)
	}
	b, e := os.ReadFile(filepath.Join(active, "SKILL.md"))
	if e != nil {
		return false, e
	}
	i := bytes.LastIndex(b, []byte(marker))
	if i < 0 || !hmac.Equal(b, signed(bytes.Clone(b[:i]), key)) {
		return false, fmt.Errorf("preserving another or modified skill at %s", active)
	}
	return true, nil
}
func ownerKey(state string) ([]byte, error) {
	p := filepath.Join(state, "owner.key")
	b, e := os.ReadFile(p)
	if e == nil {
		if len(b) != 32 {
			return nil, errors.New("invalid installer ownership record")
		}
		return b, nil
	}
	if !errors.Is(e, os.ErrNotExist) {
		return nil, e
	}
	b = make([]byte, 32)
	if _, e = rand.Read(b); e != nil {
		return nil, e
	}
	f, e := os.OpenFile(p, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		return nil, e
	}
	_, e = f.Write(b)
	closeErr := f.Close()
	if e != nil {
		return nil, e
	}
	return b, closeErr
}
func install(o options, choice string) error {
	base := o.userDir
	var e error
	if base == "" {
		base, e = os.UserHomeDir()
		if e != nil {
			return e
		}
	}
	base, e = filepath.Abs(base)
	if e != nil {
		return e
	}
	active := filepath.Join(base, ".agents", "skills", name)
	state := filepath.Join(base, ".csi-training")
	zero := strings.EqualFold(choice, "K0") || choice == "0"
	if zero {
		if _, e = os.Lstat(active); errors.Is(e, os.ErrNotExist) {
			fmt.Println("K0: no CSI skill active.")
			return nil
		}
	}
	if e = os.MkdirAll(state, 0700); e != nil {
		return e
	}
	lock, e := lockState(filepath.Join(state, "switch.lock"))
	if e != nil {
		return e
	}
	defer lock.Close()
	key, e := ownerKey(state)
	if e != nil {
		return e
	}
	exists, e := owned(active, key)
	if e != nil {
		return e
	}
	parent := filepath.Dir(active)
	stageParent := filepath.Dir(parent)
	if zero {
		if exists {
			stage, e := os.MkdirTemp(stageParent, ".csi-retire-")
			if e != nil {
				return e
			}
			defer os.RemoveAll(stage)
			if e = os.Rename(active, filepath.Join(stage, "owned-skill")); e != nil {
				return e
			}
		}
		fmt.Println("K0: no CSI skill active.")
		return nil
	}
	data, e := verifiedIndex(o)
	if e != nil {
		return e
	}
	id, c, e := resolve(data, choice)
	if e != nil {
		return e
	}
	dir, roots, e := snapshot(o, state, c)
	if e != nil {
		return e
	}
	if e = os.MkdirAll(parent, 0755); e != nil {
		return e
	}
	stage, e := os.MkdirTemp(stageParent, ".csi-install-")
	if e != nil {
		return e
	}
	defer os.RemoveAll(stage)
	newFile := filepath.Join(stage, "SKILL.md")
	if e = os.WriteFile(newFile, signed(wrapper(id, dir, roots), key), 0644); e != nil {
		return e
	}
	if exists {
		e = os.Rename(newFile, filepath.Join(active, "SKILL.md"))
	} else {
		e = publishDirectory(stage, active)
	}
	if e == nil {
		fmt.Printf("%s -> %s\nCodex: $%s\n", id, active, name)
	}
	return e
}
func line(r *bufio.Reader, prompt string) (string, error) {
	fmt.Print(prompt)
	s, e := r.ReadString('\n')
	if e != nil && len(s) == 0 {
		return "", e
	}
	return strings.TrimSpace(s), nil
}
func main() {
	var o options
	flag.StringVar(&o.offline, "offline", "", "directory containing the release manifest and checkpoint archives")
	flag.StringVar(&o.userDir, "user-dir", "", "alternate agent user profile, for isolated testing")
	flag.Parse()
	interactive := flag.NArg() == 0
	reader := bufio.NewReader(os.Stdin)
	run := func() error {
		if flag.NArg() > 1 {
			return errors.New("pass one checkpoint: K0-K76, legacy-K20, or flat-K20")
		}
		v := flag.Arg(0)
		if interactive {
			var e error
			v, e = line(reader, "Checkpoint K0-K76, legacy-K20, flat-K20 (Enter=K76): ")
			if e != nil {
				return e
			}
			if v == "" {
				v = "K76"
			}
		}
		return install(o, v)
	}
	err := run()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Installation:", err)
	}
	if err != nil {
		os.Exit(1)
	}
}

// Claim a new directory without replacing an unrelated skill, even if a different installer arrives concurrently.
func publishDirectory(stage, active string) error {
	if e := os.Mkdir(active, 0755); e != nil {
		return e
	}
	if e := os.Link(filepath.Join(stage, "SKILL.md"), filepath.Join(active, "SKILL.md")); e != nil {
		os.Remove(active) // Remove only our newly claimed directory if it is still empty.
		return e
	}
	return nil
}
