import pathlib
import sys
import trace
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = ROOT / "chat_npu_13.py"
MIN_COVERAGE = 90.0

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def executable_lines(path: pathlib.Path) -> set[int]:
    source = path.read_text(encoding="utf-8")
    code = compile(source, str(path), "exec")

    def walk_code_objects(obj: types.CodeType) -> set[int]:
        out = {lineno for _, _, lineno in obj.co_lines() if isinstance(lineno, int)}
        for const in obj.co_consts:
            if isinstance(const, types.CodeType):
                out |= walk_code_objects(const)
        return out

    return walk_code_objects(code)


def main() -> int:
    tracer = trace.Trace(count=True, trace=False)
    runner = unittest.TextTestRunner(verbosity=2)

    def discover_and_run():
        loader = unittest.defaultTestLoader
        suite = loader.discover(str(ROOT / "tests"), pattern="test_*.py")
        return runner.run(suite)

    result = tracer.runfunc(discover_and_run)
    if not result.wasSuccessful():
        print("\nTests failed. Coverage not evaluated.")
        return 1

    counts = tracer.results().counts
    target_resolved = TARGET.resolve()
    executed = {
        lineno
        for (filename, lineno), hit_count in counts.items()
        if pathlib.Path(filename).resolve() == target_resolved and hit_count > 0
    }
    executable = executable_lines(TARGET)
    covered = len(executed & executable)
    total = len(executable)
    pct = (covered / total * 100.0) if total else 100.0

    print(f"\nCoverage (approx, stdlib trace) for {TARGET.name}: {pct:.2f}% ({covered}/{total})")
    if pct < MIN_COVERAGE:
        print(f"Coverage check failed: minimum is {MIN_COVERAGE:.1f}%")
        return 1

    print(f"Coverage check passed: >= {MIN_COVERAGE:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
