// Prefix the site's base path onto root-relative links typed in page content.
//
// Pages link to each other as /results/, /limits/ and so on, which is right at a domain root and wrong for a
// GitHub project site (https://user.github.io/repo/). Astro's `base` option rewrites its own links and assets
// but not links typed in Markdown/MDX, so after the build this pass adds the base to every root-relative
// href/src that lacks it. With no base (SITE_BASE unset or "/") it does nothing.
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

export default function baseLinks({ base }) {
	const prefix = (base || '/').replace(/\/+$/, '');
	return {
		name: 'base-links',
		hooks: {
			'astro:build:done': ({ dir }) => {
				if (!prefix) return;
				let files = 0;
				let links = 0;
				for (const file of walk(fileURLToPath(dir))) {
					if (!file.endsWith('.html')) continue;
					const html = fs.readFileSync(file, 'utf8');
					const out = html.replace(/((?:href|src)=")(\/(?!\/)[^"]*)"/g, (match, attr, value) => {
						if (value === prefix || value.startsWith(prefix + '/')) return match;
						links += 1;
						return `${attr}${prefix}${value}"`;
					});
					if (out !== html) {
						fs.writeFileSync(file, out);
						files += 1;
					}
				}
				console.log(`[base-links] prefixed ${links} link(s) in ${files} page(s) with ${prefix}`);
			},
		},
	};
}
