import importlib.util
import mmap
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).parent
SCRIPT = ROOT / "probe_cuda_fatbin_library_load.py"
sys.path.insert(0, str(ROOT))
from probe_cuda_fatbin_cubin_load import FATBIN_FILE_HEADER, FATBIN_KIND_ELF, FATBIN_MAGIC, FATBIN_VERSION

SPEC = importlib.util.spec_from_file_location("probe_cuda_fatbin_library_load", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def fatbin(sm_version: int) -> bytes:
    entry = FATBIN_FILE_HEADER.pack(
        FATBIN_KIND_ELF,
        0x101,
        FATBIN_FILE_HEADER.size,
        0,
        0,
        0,
        0,
        0,
        sm_version,
        0,
        0,
        0,
        0,
        0,
    )
    return module.FATBIN_HEADER.pack(
        FATBIN_MAGIC,
        FATBIN_VERSION,
        module.FATBIN_HEADER.size,
        len(entry),
    ) + entry


class ExplicitFatbinOffsetTests(unittest.TestCase):
    def test_selects_the_declared_offset_when_files_size_is_ambiguous(self):
        first = fatbin(90)
        second_offset = 0x100
        second = fatbin(120)
        blob = first + b"\0" * (second_offset - len(first)) + second
        with tempfile.NamedTemporaryFile() as source:
            source.write(blob)
            source.flush()
            with open(source.name, "rb") as mapped_source:
                mapped = mmap.mmap(mapped_source.fileno(), 0, access=mmap.ACCESS_READ)
                try:
                    offset, entries = module.locate_fatbin_at_offset(
                        mapped,
                        second_offset,
                        len(second) - module.FATBIN_HEADER.size,
                    )
                finally:
                    mapped.close()
        self.assertEqual(offset, second_offset)
        self.assertEqual(module.entry_shape(entries), [("elf", 120)])

    def test_rejects_an_offset_that_is_not_a_complete_declared_fatbin(self):
        payload = fatbin(90)
        with tempfile.NamedTemporaryFile() as source:
            source.write(payload)
            source.flush()
            with open(source.name, "rb") as mapped_source:
                mapped = mmap.mmap(mapped_source.fileno(), 0, access=mmap.ACCESS_READ)
                try:
                    with self.assertRaisesRegex(module.ProbeFailure, "fatbin offset"):
                        module.locate_fatbin_at_offset(
                            mapped,
                            1,
                            len(payload) - module.FATBIN_HEADER.size,
                        )
                finally:
                    mapped.close()


if __name__ == "__main__":
    unittest.main()
