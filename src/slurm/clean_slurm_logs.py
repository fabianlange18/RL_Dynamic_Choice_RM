import re
from pathlib import Path

def clean_file(path: Path, pattern):
    text = path.read_text()
    lines = text.splitlines()
    new_lines = [l for l in lines if not pattern.search(l)]
    if len(new_lines) != len(lines):
        path.write_text('\n'.join(new_lines) + ("\n" if new_lines else ""))
        return True
    return False


def main():
    log_dir = Path('src/slurm/logs')
    if not log_dir.exists():
        print('Log directory not found:', log_dir)
        return
    pattern = re.compile(r"slurm_curl_request: curl_easy_perform failed to send data|error: send data failed")
    changed = []
    for p in sorted(log_dir.glob('*.err')):
        try:
            if clean_file(p, pattern):
                changed.append(str(p))
        except Exception as e:
            print('Failed to clean', p, e)
    print('Cleaned files:', len(changed))
    for c in changed:
        print(' -', c)

if __name__ == '__main__':
    main()
