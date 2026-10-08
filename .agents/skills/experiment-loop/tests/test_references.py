"""Fast standard-library checks that the skill's procedures point at real files."""
from pathlib import Path
import re
import unittest

SKILL = Path(__file__).resolve().parents[1]
ROOT = SKILL.parents[2]
DOCUMENTS = [SKILL / 'SKILL.md', *sorted((SKILL / 'references').glob('*.md'))]
LINK = re.compile(r'\]\(([^)#\s]+)(?:#[^)]*)?\)')
# Backticked repository paths, e.g. `workflow/networks/opt_mlp.py`.
PATH = re.compile(r'`((?:[\w.-]+/)+[\w.-]+\.(?:py|md|json))`')


class ReferenceTests(unittest.TestCase):
    def test_skill_frontmatter_names_and_describes_the_skill(self):
        text = (SKILL / 'SKILL.md').read_text(encoding='utf-8')
        header = text.split('---')[1]
        self.assertIn('name: experiment-loop', header)
        self.assertRegex(header, r'description: \S')

    def test_relative_links_resolve(self):
        for document in DOCUMENTS:
            for target in LINK.findall(document.read_text(encoding='utf-8')):
                if '://' in target:
                    continue
                with self.subTest(document=document.name, target=target):
                    self.assertTrue((document.parent / target).exists())

    def test_named_repository_paths_exist(self):
        for document in DOCUMENTS:
            for path in PATH.findall(document.read_text(encoding='utf-8')):
                if '<' in path or path.startswith(('campaigns/pilots/', 'results/')):
                    continue
                with self.subTest(document=document.name, path=path):
                    self.assertTrue((ROOT / path).exists())


if __name__ == '__main__':
    unittest.main()
