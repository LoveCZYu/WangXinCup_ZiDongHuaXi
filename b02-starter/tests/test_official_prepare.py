"""Official ZIP layout compatibility and extraction boundary checks."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import warnings
from zipfile import ZipFile

from b02.official import prepare, SOURCE_FILES


def write_archive(path, roots, duplicate=False):
    with ZipFile(path, 'w') as archive:
        for root in roots:
            for name in SOURCE_FILES:
                archive.writestr(f'{root}/src/{name}', f'# fixture {name}\n')
            archive.writestr(f'{root}/data/official/serve.exe', b'fixture EXE')
            archive.writestr(f'{root}/data/inputs/train_005.dat', b'fixture DAT')
            archive.writestr(f'{root}/LICENSE', 'fixture license')
            archive.writestr(f'{root}/src/.idea/workspace.xml', 'must not extract IDE metadata')
            archive.writestr(f'{root}/../../outside.txt', 'must not escape extraction root')
        if duplicate:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                archive.writestr(f'{roots[0]}/src/client.py', '# duplicate')


class OfficialPrepareTest(unittest.TestCase):
    def test_new_and_legacy_layouts_extract_runtime_and_license_only(self):
        for root in ('participant_demo', 'participant_demo_v2/participant_demo'):
            with self.subTest(root=root), tempfile.TemporaryDirectory() as folder:
                base=Path(folder); source=base/'official.zip'; target=base/'prepared'
                write_archive(source, [root])
                with contextlib.redirect_stdout(io.StringIO()):
                    prepare(source, target)
                self.assertEqual((target/'src/client.py').read_text(), '# fixture client.py\n')
                self.assertEqual((target/'LICENSE').read_text(), 'fixture license')
                self.assertTrue((target/'data/inputs/train_005.dat').is_file())
                self.assertFalse((target/'src/.idea').exists())
                self.assertFalse((base/'outside.txt').exists())

    def test_ambiguous_complete_roots_are_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder); source=base/'official.zip'; target=base/'prepared'
            write_archive(source, ['participant_demo', 'participant_demo_v2/participant_demo'])
            with self.assertRaisesRegex(ValueError, '只能包含一套'):
                prepare(source, target)
            self.assertFalse(target.exists())

    def test_duplicate_runtime_member_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder); source=base/'official.zip'; target=base/'prepared'
            write_archive(source, ['participant_demo'], duplicate=True)
            with self.assertRaisesRegex(ValueError, '重复文件'):
                prepare(source, target)
            self.assertFalse(target.exists())

    def test_existing_runtime_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder); source=base/'official.zip'; target=base/'prepared'
            write_archive(source, ['participant_demo'])
            target.mkdir(); sentinel=target/'sentinel'; sentinel.write_text('preserve')
            with self.assertRaisesRegex(ValueError, '已存在'):
                prepare(source, target)
            self.assertEqual(sentinel.read_text(), 'preserve')

    def test_incomplete_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder); source=base/'official.zip'; target=base/'prepared'
            with ZipFile(source,'w') as archive:
                archive.writestr('participant_demo/src/client.py', '# incomplete')
            with self.assertRaises(ValueError):
                prepare(source, target)
            self.assertFalse(target.exists())


if __name__=='__main__':
    unittest.main()
