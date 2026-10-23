// Build guards for the AMM site.
//
// Private preview (default): the build writes robots.txt and _headers that block indexing and
// removes the sitemap. Unfinished claims are listed as a warning.
// Public build (AMM_LAUNCH=1): the build FAILS if any rendered claim is pending, or is in-sample
// while marked requiresFinal, or if the author block is rendered while AMM_SHOW_AUTHOR is off.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

function* walk(dir) {
	for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
		const full = path.join(dir, entry.name);
		if (entry.isDirectory()) yield* walk(full);
		else yield full;
	}
}

export default function ammGuards({ launch, showAuthor }) {
	return {
		name: 'amm-guards',
		hooks: {
			'astro:build:done': ({ dir }) => {
				const root = fileURLToPath(dir);
				const problems = [];
				const unfinished = new Set();
				for (const file of walk(root)) {
					if (!file.endsWith('.html')) continue;
					const html = fs.readFileSync(file, 'utf8');
					const rel = path.relative(root, file);
					for (const m of html.matchAll(/data-claim-id="([^"]+)"[^>]*data-claim-status="([^"]+)"[^>]*data-claim-final-required="([^"]+)"/g)) {
						const [, id, status, required] = m;
						if (status === 'pending' || (status === 'dev' && required === 'true')) {
							problems.push(`${rel}: claim "${id}" is ${status}${required === 'true' ? ' and must be final' : ''}`);
							unfinished.add(id);
						}
					}
					if (!showAuthor && html.includes('data-author-block')) {
						problems.push(`${rel}: author block rendered while AMM_SHOW_AUTHOR is off`);
					}
					if (launch && /<meta[^>]+name="robots"[^>]+noindex/i.test(html)) {
						problems.push(`${rel}: noindex meta present in a public build`);
					}
				}

				if (launch) {
					if (problems.length) {
						throw new Error(
							`AMM_LAUNCH build refused: ${problems.length} problem(s)\n  ` + problems.join('\n  '),
						);
					}
					fs.writeFileSync(path.join(root, 'robots.txt'), 'User-agent: *\nAllow: /\n');
					console.log('[amm-guards] public build OK: every rendered claim is final.');
				} else {
					fs.writeFileSync(path.join(root, 'robots.txt'), 'User-agent: *\nDisallow: /\n');
					fs.writeFileSync(path.join(root, '_headers'), '/*\n  X-Robots-Tag: noindex, nofollow\n');
					for (const file of fs.readdirSync(root)) {
						if (/^sitemap.*\.xml$/.test(file)) fs.rmSync(path.join(root, file));
					}
					const authorProblems = problems.filter((p) => p.includes('author block'));
					if (authorProblems.length) throw new Error(authorProblems.join('\n'));
					console.log(
						`[amm-guards] private preview: noindex written; ${unfinished.size} unfinished claim(s) shown (allowed here): ${[...unfinished].join(', ') || 'none'}`,
					);
				}
			},
		},
	};
}
