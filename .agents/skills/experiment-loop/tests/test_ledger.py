"""Fast standard-library tests; never run research experiments."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'ledger.py'
spec = importlib.util.spec_from_file_location('campaign_ledger', SCRIPT)
ledger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ledger)


def put(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def note(path, meta):
    put(path, '---\n' + ''.join(k + ': ' + json.dumps(v) + '\n' for k, v in meta.items()) + '---\n\nBody.\n')


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'campaign'
        put(self.root / 'README.md', '# Question\n')
        put(self.root / 'ideas.md', 'H-001\n')
        put(self.root / 'explorations/X-001-idea.md', '# Reasoning\n')
        self.hp = self.root / 'hypotheses/H-001-claim.md'
        note(self.hp, dict(id='H-001', title='Claim'))
        self.series = self.root / 'series/001-test'
        put(self.series / 'README.md', '# Series\n')
        (self.series / 'results').mkdir()
        self.ep = self.series / 'experiments/exp-001-test.md'
        note(self.ep, dict(id='exp-001', title='Test', status='planned', hypotheses=['H-001']))

    def call(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return ledger.main(list(args))

    def result(self, evidence='validated-local', verdicts=None):
        rp = self.series / 'results/result.md'
        note(rp, dict(experiment='exp-001', evidence=evidence, summary='Observation',
                      verdicts={'H-001': 'inconclusive'} if verdicts is None else verdicts))
        return rp

    def test_exploration_only_campaign(self):
        import shutil
        shutil.rmtree(self.root / 'hypotheses')
        shutil.rmtree(self.root / 'series')
        (self.root / 'ideas.md').unlink()
        self.assertIn('Campaign ledger', ledger.render(self.root))

    def test_experiment_without_hypothesis_hierarchy(self):
        import shutil
        shutil.rmtree(self.root / 'hypotheses')
        shutil.rmtree(self.root / 'explorations')
        (self.root / 'ideas.md').unlink()
        note(self.ep, dict(id='exp-001', title='Question', status='planned', hypotheses=[]))
        self.assertIn('None declared', ledger.render(self.root))

    def test_deterministic_render(self):
        self.assertEqual(ledger.render(self.root), ledger.render(self.root))

    def test_generation_and_check(self):
        self.assertEqual(self.call(str(self.root)), 0)
        self.assertEqual(self.call('--check', str(self.root)), 0)
        self.assertFalse(list(self.root.glob('.ledger-*')))

    def test_stale_check_never_writes(self):
        put(self.root / 'ledger.md', 'old')
        self.assertEqual(self.call('--check', str(self.root)), 1)
        self.assertEqual((self.root / 'ledger.md').read_text(), 'old')

    def test_missing_check_never_creates(self):
        self.assertEqual(self.call('--check', str(self.root)), 1)
        self.assertFalse((self.root / 'ledger.md').exists())

    def test_duplicate_hypothesis(self):
        note(self.hp.with_name('H-001-other.md'), dict(id='H-001', title='Other'))
        with self.assertRaises(ledger.CampaignError): ledger.render(self.root)

    def test_unknown_hypothesis(self):
        note(self.ep, dict(id='exp-001', title='Test', status='planned', hypotheses=['H-999']))
        with self.assertRaises(ledger.CampaignError): ledger.render(self.root)

    def test_completed_requires_result(self):
        note(self.ep, dict(id='exp-001', title='Test', status='complete', hypotheses=['H-001']))
        with self.assertRaises(ledger.CampaignError): ledger.render(self.root)
        self.result()
        self.assertIn('complete', ledger.render(self.root))

    def test_import_cannot_claim_fresh_validation(self):
        note(self.ep, dict(id='exp-001', title='Test', status='imported', hypotheses=['H-001']))
        self.result()
        with self.assertRaises(ledger.CampaignError): ledger.render(self.root)
        self.result('imported-summary')
        self.assertIn('imported-summary', ledger.render(self.root))

    def test_result_only_declared_hypotheses(self):
        self.result(verdicts={'H-999': 'supports'})
        with self.assertRaises(ledger.CampaignError): ledger.render(self.root)

    def test_idea_board_coverage(self):
        put(self.root / 'ideas.md', 'H-999\n')
        with self.assertRaises(ledger.CampaignError): ledger.render(self.root)

    def test_duplicate_metadata_key(self):
        put(self.hp, '---\nid: "H-001"\nid: "H-001"\ntitle: "Claim"\n---\n')
        with self.assertRaises(ledger.CampaignError): ledger.render(self.root)

    def test_unknown_metadata_key(self):
        note(self.hp, dict(id='H-001', title='Claim', status='accepted'))
        with self.assertRaises(ledger.CampaignError): ledger.render(self.root)

    def test_duplicate_experiment_across_series(self):
        other = self.root / 'series/002-other'
        put(other / 'README.md', '# Other\n')
        (other / 'results').mkdir()
        note(other / 'experiments/exp-001-other.md', dict(id='exp-001', title='Other', status='planned', hypotheses=['H-001']))
        with self.assertRaises(ledger.CampaignError): ledger.render(self.root)

    def test_conflicting_verdicts_remain_visible(self):
        self.result(verdicts={'H-001': 'supports'})
        note(self.series / 'results/correction.md', dict(experiment='exp-001', evidence='validated-local', summary='Correction', verdicts={'H-001': 'contradicts'}))
        output = ledger.render(self.root)
        self.assertIn('supports', output)
        self.assertIn('contradicts', output)

    def test_symlink_is_rejected(self):
        (self.root / 'ledger.md').symlink_to(self.root / 'README.md')
        with self.assertRaises(ledger.CampaignError): ledger.render(self.root)

    def test_bad_metadata_type_is_actionable(self):
        note(self.ep, dict(id='exp-001', title='Test', status='planned', hypotheses=[{}]))
        with self.assertRaises(ledger.CampaignError): ledger.render(self.root)

    def test_validate_all_before_write(self):
        other = Path(self.temp.name) / 'missing'
        self.assertEqual(self.call(str(self.root), str(other)), 2)
        self.assertFalse((self.root / 'ledger.md').exists())


if __name__ == '__main__':
    unittest.main()
