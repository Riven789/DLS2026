"""Create an original synthetic teaching corpus. It is not a benchmark."""
from pathlib import Path


def write_corpus(path):
    subjects = ['The atom', 'The molecule', 'The crystal', 'The magnet', 'The detector', 'The plasma', 'The protein', 'The battery']
    verbs = ['stores', 'releases', 'absorbs', 'transfers', 'measures', 'changes']
    objects = ['energy', 'heat', 'charge', 'light', 'momentum', 'a signal']
    docs = [f'{s} {v} {o}. Scientists compare the model with data.' for s in subjects for v in verbs for o in objects]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n\n'.join(docs) + '\n', encoding='utf-8')
    return len(docs)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/science.txt')
    args = parser.parse_args()
    print(f'Wrote {write_corpus(args.out)} synthetic documents to {args.out}')
