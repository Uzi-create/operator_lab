import subprocess
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import operators.build as build
import os


class AtomicBuildTest(unittest.TestCase):
    def test_platform_commands_and_filenames(self):
        for platform, name, linker in [('win32','operators.dll','-shared'),
                                        ('linux','liboperators.so','-shared'),
                                        ('darwin','liboperators.dylib','-dynamiclib')]:
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                def link(command,check,cwd):
                    self.assertIn(linker,command)
                    self.assertIn('-static' if platform=='win32' else '-fPIC',command)
                    self.assertFalse(Path(command[-1]).is_absolute())
                    (Path(cwd)/command[-1]).write_bytes(b'library')
                with patch.object(build,'ROOT',root), patch.object(build.sys,'platform',platform), \
                     patch.object(build.subprocess,'run',side_effect=link):
                    self.assertEqual(build.build().name,name)

    def test_windows_compiler_path_with_spaces(self):
        with patch.object(build.sys,'platform','win32'), \
             patch.dict(os.environ,{'CXX':'"C:/Compiler Tools/bin/g++.exe" -march=x86-64'}):
            self.assertEqual(build.compiler_command(),['C:/Compiler Tools/bin/g++.exe','-march=x86-64'])

    def test_failed_link_preserves_existing_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);library=root/build.library_name()
            library.write_bytes(b'previous-library')
            def fail(command,check,cwd):
                (Path(cwd)/command[-1]).write_bytes(b'incomplete')
                raise subprocess.CalledProcessError(1,command)
            with patch.object(build,'ROOT',root),patch.object(build.subprocess,'run',side_effect=fail):
                with self.assertRaises(subprocess.CalledProcessError):build.build()
            self.assertEqual(library.read_bytes(),b'previous-library')
            self.assertEqual(list(root.iterdir()),[library])

    def test_success_replaces_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def link(command,check,cwd):(Path(cwd)/command[-1]).write_bytes(b'complete-library')
            with patch.object(build,'ROOT',root),patch.object(build.subprocess,'run',side_effect=link):
                output=build.build()
            self.assertEqual(output.read_bytes(),b'complete-library')
            self.assertEqual(list(root.iterdir()),[output])

if __name__=='__main__':unittest.main()
