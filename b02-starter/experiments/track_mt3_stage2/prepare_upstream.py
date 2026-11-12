"""Create an isolated 3D fork of the pinned CNSF source; never edit stage 1."""
import argparse
import difflib
from pathlib import Path
import shutil
import subprocess

PIN = '22c3a19c15df6c560eaadc74462b9b632ede2602'


def prepare(source, destination):
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != PIN:
        raise ValueError(f'Expected upstream {PIN}, got {commit}')
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', '--quiet', 'HEAD',
                    '--', 'track_mt3', 'configs', 'LICENSE'], check=True)
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('track_mt3', 'configs'):
        shutil.copytree(source / name, destination / name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(source / 'LICENSE', destination / 'LICENSE')
    diffs = []

    def change(name, pairs):
        path = destination / 'track_mt3' / name
        before = path.read_text()
        after = before
        for old, new in pairs:
            if old not in after:
                raise ValueError(f'Patch context absent in {name}: {old}')
            after = after.replace(old, new)
        path.write_text(after)
        diffs.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                        fromfile=name, tofile=name))

    change('models/normalization.py', [
        ('x_min, x_max, y_min, y_max = field_of_view',
         'bounds = torch.tensor(field_of_view, dtype=torch.float32).reshape(-1, 2)'),
        ('torch.tensor([x_max - x_min, y_max - y_min], dtype=torch.float32)', 'bounds[:, 1] - bounds[:, 0]'),
        ('torch.tensor([(x_max + x_min) / 2.0, (y_max + y_min) / 2.0], dtype=torch.float32)', 'bounds.mean(dim=1)'),
    ])
    # Preserve the pretrained 2D Fourier layout; add a fixed elevation signature.
    change('models/spatial_encoding.py', [
        ('if hidden_dim % (2 * coordinate_dim):', 'self.input_coordinate_dim = coordinate_dim\n        coordinate_dim = 2 if coordinate_dim == 3 else coordinate_dim\n        if hidden_dim % (2 * coordinate_dim):'),
        ('if unit_positions.shape[-1] != self.coordinate_dim:',
         'if unit_positions.shape[-1] == 3 and self.input_coordinate_dim == 3:\n            base = self.forward(unit_positions[..., :2])\n            elevation = unit_positions[..., 2:3].expand(*unit_positions.shape[:-1], 2)\n            return base + 0.1 * self.forward(elevation)\n        if unit_positions.shape[-1] != self.coordinate_dim:'),
    ])
    change('models/qtm.py', [('source_dim = 3 if', 'source_dim = config.output_dim + 1 if')])
    change('tracking/track_state.py', [('torch.empty((0, 2)', 'torch.empty((0, 3)'),
                                       ('(size, 2)', '(size, 3)'), ('shape [N, 2]', 'shape [N, 3]')])
    change('data/window.py', [('max_measurements, 2)', 'max_measurements, windows[0].measurements.shape[-1])')])
    change('models/track_mt3.py', [
        ('torch.zeros((1, 2), device=self.device)', 'torch.zeros((1, self.config.model.measurement_dim), device=self.device)'),
        ('guaranteed = index == 0 and time_index < 2',
         'guaranteed = torch.distributed.is_initialized() and index == 0 and time_index < 2'),
        ('                assignment = alignments[index](',
         '                if hasattr(criterion, "set_window"):\n                    criterion.set_window(window)\n                assignment = alignments[index]('),
    ])
    # Unknown B02 measurements map to -2, unlike known clutter (-1). Exclude both
    # ends of unknown pairs from the supervised contrastive denominator as well.
    change('models/contrastive.py', [])
    track = destination/'track_mt3/models/track_mt3.py'
    text = track.read_text()
    needle = 'self.contrastive_classifier(memory, memory_mask)'
    if needle not in text:
        raise ValueError('Contrastive call changed upstream')
    change('models/track_mt3.py', [(needle,
        'self.contrastive_classifier(memory, memory_mask | (measurement_ids.to(self.device) == -2))')])
    (destination/'b02-stage2.patch').write_text(''.join(diffs))
    (destination/'UPSTREAM_COMMIT').write_text(commit+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    prepare(args.source, args.destination)
